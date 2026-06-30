"""
Convert ARX robot data (JSON + videos) to LeRobot format using zero-copy mode.

Zero-copy mode:
- Inherits LeRobotDataset and overrides save_episode() to directly copy videos
- No video decoding/re-encoding (10x+ faster)
- Features use dtype="video"
- add_frame() only receives non-image data
- save_episode() receives video paths for direct copying

Usage:
    # Convert with video copying (recommended)
    uv run examples/arx/convert_arx_data_to_lerobot.py \
        --data_dirs /path/to/data \
        --output_repo_id arx_dataset

    # True zero-copy (reference original videos)
    uv run examples/arx/convert_arx_data_to_lerobot.py \
        --data_dirs /path/to/data \
        --output_repo_id arx_dataset \
        --copy_videos False

    # Resume interrupted conversion
    uv run examples/arx/convert_arx_data_to_lerobot.py \
        --data_dirs /path/to/data \
        --output_repo_id arx_dataset \
        --resume

Note: All joint data is assumed to be in radians (no conversion needed).
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from lerobot.common.datasets.compute_stats import compute_episode_stats
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ========================================================================
# Custom ARX Dataset with zero-copy video support
# ========================================================================


class ArxDataset(LeRobotDataset):
    """
    Custom LeRobot dataset that supports zero-copy video conversion.

    This class overrides save_episode() to directly copy video files
    instead of re-encoding them, following the pattern from agi_convert_to_lerobot.py.

    """

    def save_episode(
        self, episode_data: dict | None = None, videos: dict | None = None
    ) -> None:
        """
        Save episode with direct video file copying (zero-copy mode).

        Args:
            task: Task description/instruction
            episode_data: Optional episode data (if None, uses episode_buffer)
            videos: Dictionary mapping camera names to video file paths
                   e.g., {"wrist_right": Path("video.mp4"), ...}
        """
        # Use provided episode_data or fall back to episode_buffer
        episode_buffer = episode_data if episode_data else self.episode_buffer

        episode_length = episode_buffer.pop("size")
        episode_index = episode_buffer["episode_index"]
        #fit for task
        tasks = episode_buffer.pop("task")

        # Add new tasks to the tasks dictionary
        episode_tasks = list(set(tasks))
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        if episode_index != self.meta.total_episodes:
            raise NotImplementedError(
                "You might have manually provided the episode_buffer with an episode_index that doesn't "
                "match the total number of episodes in the dataset. This is not supported for now."
            )

        if episode_length == 0:
            raise ValueError(
                "You must add one or several frames with `add_frame` before calling `save_episode`."
            )

        # task_index = self.meta.get_task_index(task)

        # In zero-copy mode, video keys are NOT in episode_buffer yet (will be added later)
        # Only validate that non-video feature keys are present
        expected_keys = {
            k for k, ft in self.features.items()
            if ft["dtype"] not in ["image", "video"]
        }
        buffer_keys = set(episode_buffer.keys())

        # Check for missing required keys
        missing_keys = expected_keys - buffer_keys
        if missing_keys:
            raise ValueError(
                f"Missing required keys in episode_buffer: {missing_keys}. "
                f"Expected: {expected_keys}, Got: {buffer_keys}"
            )

        # Check for unexpected keys (except video keys which will be added later)
        unexpected_keys = buffer_keys - expected_keys - set(self.meta.video_keys)
        # if unexpected_keys:
        #     print("*****************************",unexpected_keys)
            # raise ValueError(
            #     f"Unexpected keys in episode_buffer: {unexpected_keys}. "
            #     f"These keys are not in the dataset features."
            # )

        # Prepare episode buffer data
        for key, ft in self.features.items():
            if key == "index":
                episode_buffer[key] = np.arange(
                    self.meta.total_frames, self.meta.total_frames + episode_length
                )
            elif key == "episode_index":
                episode_buffer[key] = np.full((episode_length,), episode_index)
            # elif key == "task_index":
            #     episode_buffer[key] = np.full((episode_length,), task_index)
            elif ft["dtype"] in ["image", "video"]:
                # Skip image/video features - will be handled separately
                continue
            elif len(ft["shape"]) == 1 and ft["shape"][0] == 1:
                episode_buffer[key] = np.array(episode_buffer[key], dtype=ft["dtype"])
            elif len(ft["shape"]) == 1 and ft["shape"][0] > 1:
                episode_buffer[key] = np.stack(episode_buffer[key])
            else:
                raise ValueError(f"Unsupported feature shape: {key} -> {ft}")



        ep_calc_stats = {}
        for key, value in episode_buffer.items():
            if key in self.features.keys() and self.features[key]["dtype"] not in ["video", "image"] and key != "task_index":
                ep_calc_stats[key] = value
        ep_stats = compute_episode_stats(ep_calc_stats, self.features)
        # Wait for any pending image writes
        self._wait_image_writer()

        # 🎯 Core zero-copy logic: directly copy video files
        if videos is not None:
            for key in self.meta.video_keys:
                # Get destination path from LeRobot metadata
                video_path = self.root / self.meta.get_video_file_path(episode_index, key)
                episode_buffer[key] = video_path
                video_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract camera name from key (e.g., "observation.images.wrist_right" -> "wrist_right")
                camera_name = key.split(".")[-1]

                if camera_name not in videos:
                    raise ValueError(
                        f"Missing video for camera '{camera_name}'. "
                        f"Expected one of: {list(videos.keys())}"
                    )

                # ✨ Direct copy - no video decoding/encoding!
                shutil.copy2(videos[camera_name], video_path)

        # Save episode table
        self._save_episode_table(episode_buffer, episode_index)

        # Save episode metadata
        self.meta.save_episode(episode_index, episode_length, task, ep_stats)

        # Reset episode buffer if not using external episode_data
        if episode_data is None:
            self.episode_buffer = self.create_episode_buffer()

        self.consolidated = False

    def add_frame(self, frame: dict) -> None:
        """
        Add a frame to the current episode.

        Note: For zero-copy mode, frame should NOT contain image data.
        Images will be handled separately via the videos parameter in save_episode().
        """
        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = (
            frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        )
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key not in self.features:
                if key == 'task':
                    self.episode_buffer[key].append(frame[key])
                    continue
                raise ValueError(f"Unknown feature key: {key}")
            item = (
                frame[key].numpy()
                if isinstance(frame[key], torch.Tensor)
                else frame[key]
            )
            self.episode_buffer[key].append(item)

        self.episode_buffer["size"] += 1


def find_all_json_files(parent_dirs: List[str]) -> List[Path]:
    """Recursively find all JSON files in parent directories."""
    json_files = []
    for parent_dir in parent_dirs:
        parent_path = Path(parent_dir)
        if not parent_path.exists():
            logger.warning(f"Directory does not exist: {parent_dir}")
            continue

        # Find all JSON files recursively
        found_files = list(parent_path.rglob("*.json"))
        json_files.extend(found_files)
        logger.info(f"Found {len(found_files)} JSON files in {parent_dir}")

    return sorted(json_files)


def load_video_info(video_path: str) -> tuple[int, int, int, int]:
    """
    Get video metadata without loading frames.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (total_frames, width, height, fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    cap.release()
    return total_frames, width, height, fps


def get_video_paths(json_path: Path, data: dict) -> Dict[str, Path]:
    """
    Extract and validate video paths from JSON data.

    Args:
        json_path: Path to JSON file
        data: Parsed JSON data

    Returns:
        Dictionary mapping camera names to absolute video paths
    """
    video_dir = json_path.parent
    video_paths = {}

    for camera_name, video_info_list in data["observations"].items():
        video_info = video_info_list[0]  # Take first video segment
        video_path = video_info["path"]

        # Handle relative paths
        if not Path(video_path).is_absolute():
            video_path = video_dir / video_path
        else:
            # Convert string to Path for manipulation
            video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        video_paths[camera_name] = video_path

    return video_paths


def convert_episode_copy_video(
    json_path: Path,
    dataset: ArxDataset,
    data: dict,
) -> bool:
    """
    Convert episode using zero-copy mode (copy videos without re-encoding).

    Following the pattern from agi_convert_to_lerobot.py:
    - Only add non-image data to frames via add_frame()
    - Pass video paths separately to save_episode() for direct copying

    Args:
        json_path: Path to JSON file
        dataset: ArxDataset instance
        data: Parsed JSON data

    Returns:
        True if successful
    """
    num_frames = data["num_frames"]
    instruction = data["instruction"]["general"]["instruction"][0]
    proprios = data["proprios"]
    actions = data["actions"]

    # Get video paths from original data
    video_paths = get_video_paths(json_path, data)

    # Determine final video paths based on copy_videos flag
    # Use original video paths directly (true zero-copy)
    final_video_paths = video_paths

    # Verify frame counts and determine actual usable frames
    expected_frames = num_frames
    for camera_name, video_path in final_video_paths.items():
        video_info = data["observations"][camera_name][0]
        expected_range = video_info["end"] - video_info["start"]
        total_frames, _, _, _ = load_video_info(str(video_path))

        if total_frames < expected_range:
            logger.warning(
                f"Video {camera_name} has {total_frames} frames, "
                f"expected at least {expected_range}"
            )
            expected_frames = min(expected_frames, total_frames)

    # 🎯 Key difference: Only add non-image data via add_frame()
    # Images will be handled by save_episode() via direct video copying
    for frame_idx in range(expected_frames):
        # Construct state and action
        state = np.concatenate(
            [
                np.array(proprios["left_arm_joint"][frame_idx], dtype=np.float32),
                np.array(proprios["left_gripper_pos"][frame_idx], dtype=np.float32),
                np.array(proprios["right_arm_joint"][frame_idx], dtype=np.float32),
                np.array(proprios["right_gripper_pos"][frame_idx], dtype=np.float32),
            ]
        )

        actions_concat = np.concatenate(
            [
                np.array(actions["left_arm_joint"][frame_idx], dtype=np.float32),
                np.array(actions["left_gripper_pos"][frame_idx], dtype=np.float32),
                np.array(actions["right_arm_joint"][frame_idx], dtype=np.float32),
                np.array(actions["right_gripper_pos"][frame_idx], dtype=np.float32),
            ]
        )

        # ✨ Only pass non-image data (no camera images!)
        frame_data = {
            "observation.state": state,
            "actions": actions_concat,
            "task": instruction,
        }

        dataset.add_frame(frame_data)

    # 🎯 Pass videos separately for direct copying
    dataset.save_episode(videos=final_video_paths)

    return True


def should_skip_episode(data: dict) -> tuple[bool, str]:
    """
    Check if episode should be skipped based on quality and trajectory type.

    Args:
        data: Parsed JSON data

    Returns:
        Tuple of (should_skip, reason)
    """
    # Check trajectory_type field (at root level)
    trajectory_type = data.get("trajectory_type", "").lower()
    if trajectory_type == "failure":
        return True, f"trajectory_type='{trajectory_type}'"

    # Check meta.quality field
    if "meta" in data:
        meta = data["meta"]
        quality = meta.get("quality", "").lower()
        if quality == "low":
            return True, f"meta.quality='{quality}'"

    return False, ""


def convert_episode(
    json_path: Path,
    dataset: ArxDataset,
) -> bool:
    """
    Convert a single episode from JSON format to LeRobot format (zero-copy mode).

    Args:
        json_path: Path to JSON file
        dataset: ArxDataset instance

    Returns:
        True if conversion succeeded, False if failed, None if skipped
    """
    try:
        start_time = time.time()

        # Load JSON metadata
        with open(json_path, "r") as f:
            data = json.load(f)

        # Check if episode should be skipped
        should_skip, skip_reason = should_skip_episode(data)
        if should_skip:
            logger.debug(f"⊘ Skipped {json_path.name}: {skip_reason}")
            return None  # Return None to indicate skip (not failure)

        num_frames = data["num_frames"]

        success = convert_episode_copy_video(
            json_path, dataset, data
        )

        if success:
            elapsed = time.time() - start_time
            fps = num_frames / elapsed if elapsed > 0 else 0
            logger.info(
                f"✓ {json_path.name}: {num_frames} frames "
                f"in {elapsed:.1f}s ({fps:.1f} fps)"
            )
        return success

    except Exception as e:
        logger.error(f"✗ Failed to convert {json_path.name}: {e}")
        return False


def main(
    data_dirs: List[str],
    output_repo_id: str,
    *,
    output_dir: str | None = None,
    fps: int = 30,
    resume: bool = False,
) -> None:
    """
    Convert ARX dataset to LeRobot format using zero-copy mode.

    Args:
        data_dirs: List of parent directories containing JSON files
        output_repo_id: Output repository ID (e.g., "arx_dataset")
        output_dir: Custom output directory. If None, uses HF_LEROBOT_HOME (~/.cache/lerobot)
        fps: Frames per second of the video data
        copy_videos: If True (default), copy videos to dataset directory. If False, reference original videos.
                    Set to False for true zero-copy (no disk duplication).
        resume: If True, skip already converted episodes
    """
    start_time = time.time()

    # Find all JSON files
    json_files = find_all_json_files(data_dirs)
    if len(json_files) == 0:
        logger.error("No JSON files found in the provided directories!")
        return

    logger.info(f"Found {len(json_files)} total JSON files to convert")

    # Output path - handle custom output directory
    if output_dir is not None:
        # Use custom output directory by temporarily modifying environment
        original_lerobot_home = os.environ.get("HF_LEROBOT_HOME")
        os.environ["HF_LEROBOT_HOME"] = str(output_dir)
        # Update module's cached path
        import lerobot.common.datasets.lerobot_dataset as lerobot_ds_module
        lerobot_ds_module.HF_LEROBOT_HOME = Path(output_dir)
        output_path = Path(output_dir) / output_repo_id
        logger.info(f"Using custom output directory: {output_path}")
    else:
        # Use default LeRobot cache directory
        output_path = HF_LEROBOT_HOME / output_repo_id
        logger.info(f"Using default LeRobot cache: {output_path}")
        original_lerobot_home = None

    # Handle resume mode
    if resume and output_path.exists():
        logger.info("Resume mode: checking for already converted episodes...")
        # For simplicity, we'll check episode count
        try:
            existing_dataset = ArxDataset(output_repo_id)
            existing_episodes = existing_dataset.num_episodes
            logger.info(f"Found {existing_episodes} existing episodes")

            if existing_episodes >= len(json_files):
                logger.info("All episodes already converted!")
                return

            # Skip already converted files
            json_files = json_files[existing_episodes:]
            logger.info(f"Resuming from episode {existing_episodes}")

        except Exception as e:
            logger.warning(f"Could not load existing dataset: {e}")
            logger.info("Starting fresh conversion")

    elif output_path.exists():
        logger.error(
            f"Dataset already exists at {output_path}. "
            "Delete it or use --resume to continue."
        )
        return

    # Load first JSON to get image dimensions
    # with open(json_files[0], "r") as f:
    #     first_data = json.load(f)

    # Get image dimensions from first video
    # first_video_info = first_data["observations"]["ego"][0]
    # video_dir = json_files[0].parent
    # video_path = first_video_info["path"]
    # if not Path(video_path).is_absolute():
    #     video_path = video_dir / video_path

    # _, image_width, image_height, _ = load_video_info(str(video_path))
    image_width = 640
    image_height = 480
    logger.info(f"Image dimensions: {image_width}x{image_height}")

    try:
        # Create or load LeRobot dataset
        if not output_path.exists():
            logger.info(f"Creating new dataset at {output_path}")

            # Define base features (state and action, always the same)
            base_features = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (14,),  # 6 + 1 + 6 + 1 joints in radians
                    "names": [
                        "left_joint_0",
                        "left_joint_1",
                        "left_joint_2",
                        "left_joint_3",
                        "left_joint_4",
                        "left_joint_5",
                        "left_gripper",
                        "right_joint_0",
                        "right_joint_1",
                        "right_joint_2",
                        "right_joint_3",
                        "right_joint_4",
                        "right_joint_5",
                        "right_gripper",
                    ],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": [
                        "left_joint_0",
                        "left_joint_1",
                        "left_joint_2",
                        "left_joint_3",
                        "left_joint_4",
                        "left_joint_5",
                        "left_gripper",
                        "right_joint_0",
                        "right_joint_1",
                        "right_joint_2",
                        "right_joint_3",
                        "right_joint_4",
                        "right_joint_5",
                        "right_gripper",
                    ],
                },
            }

            # Define camera features with dtype="video" (zero-copy mode)
            camera_features = {
                "observation.images.wrist_right": {
                    "dtype": "video",
                    "shape": (image_height, image_width, 3),
                    "names": ["height", "width", "channel"],
                    # "video_info": {
                    #     "video.fps": fps,
                    #     "video.codec": "h264",
                    #     "video.pix_fmt": "yuv420p",
                    #     "video.is_depth_map": False,
                    #     "has_audio": False,
                    # },
                },
                "observation.images.wrist_left": {
                    "dtype": "video",
                    "shape": (image_height, image_width, 3),
                    "names": ["height", "width", "channel"],
                    # "video_info": {
                    #     "video.fps": fps,
                    #     "video.codec": "h264",
                    #     "video.pix_fmt": "yuv420p",
                    #     "video.is_depth_map": False,
                    #     "has_audio": False,
                    # },
                },
                "observation.images.ego": {
                    "dtype": "video",
                    "shape": (image_height, image_width, 3),
                    "names": ["height", "width", "channel"],
                    # "video_info": {
                    #     "video.fps": fps,
                    #     "video.codec": "h264",
                    #     "video.pix_fmt": "yuv420p",
                    #     "video.is_depth_map": False,
                    #     "has_audio": False,
                    # },
                },
            }
            logger.info("Using dtype='video' with ArxDataset (zero-copy mode)")

            # Create ArxDataset for zero-copy
            dataset = ArxDataset.create(
                repo_id=output_repo_id,
                robot_type="arx_dual_arm",
                fps=fps,
                features={**camera_features, **base_features},
                # Minimal image writer threads (not used in zero-copy)
                image_writer_threads=1,
                image_writer_processes=1,
            )
        else:
            logger.info(f"Loading existing dataset from {output_path}")
            dataset = ArxDataset(output_repo_id)

        # Convert all episodes
        success_count = 0
        failed_count = 0
        skipped_count = 0
        total_frames = 0

        for json_file in tqdm(json_files, desc="Converting episodes"):
            result = convert_episode(
                json_file,
                dataset,
            )
            # break
            if result is True:
                success_count += 1
                # Count frames from JSON
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        total_frames += data["num_frames"]
                except Exception:
                    pass
            elif result is False:
                failed_count += 1
            elif result is None:
                skipped_count += 1

        # Summary
        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Conversion complete!")
        logger.info(f"  Total files: {len(json_files)}")
        logger.info(f"  Converted: {success_count} episodes")
        logger.info(f"  Skipped: {skipped_count} episodes (low quality or failed trajectories)")
        logger.info(f"  Failed: {failed_count} episodes (errors)")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        if total_frames > 0:
            logger.info(f"  Average: {total_frames/elapsed:.1f} fps")
        logger.info(f"  Output: {output_path}")
        logger.info(f"{'='*60}\n")

    finally:
        # Restore original environment if it was modified
        if output_dir is not None:
            if original_lerobot_home is not None:
                os.environ["HF_LEROBOT_HOME"] = original_lerobot_home
            else:
                os.environ.pop("HF_LEROBOT_HOME", None)
            # Restore module's cached value
            import lerobot.common.datasets.lerobot_dataset as lerobot_ds_module
            lerobot_ds_module.HF_LEROBOT_HOME = Path(original_lerobot_home) if original_lerobot_home else HF_LEROBOT_HOME


if __name__ == "__main__":
    tyro.cli(main)
