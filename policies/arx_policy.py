"""ARX dual-arm robot policy transforms."""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_arx_example() -> dict:
    """Creates a random input example for the ARX policy."""
    return {
        "state": np.ones((14,)),  # 6 left arm + 6 right arm + 1 left gripper + 1 right gripper
        "images/ego": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "images/wrist_right": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "images/wrist_left": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ArxInputs(transforms.DataTransformFn):
    """Inputs for the ARX dual-arm policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14] - [left_arm_joints(6), right_arm_joints(6), left_gripper(1), right_gripper(1)]
    - actions: [action_horizon, 14]

    All joints are in radians.
    Gripper range: [-3, 0] where -3 is fully open, 0 is fully closed.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    # model_type: _model.ModelType
    

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["images/ego"])
        wrist_left_image = _parse_image(data["images/wrist_left"])
        wrist_right_image = _parse_image(data["images/wrist_right"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_left_image,
                "right_wrist_0_rgb": wrist_right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # Normalize gripper values from [-3, 0] to [0, 1]
            actions = _normalize_gripper_actions(actions)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class ArxOutputs(transforms.DataTransformFn):
    """Outputs for the ARX dual-arm policy."""

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims
        actions = np.asarray(data["actions"][:, :14])
        # Denormalize gripper values from [0, 1] back to [-3, 0]
        actions = _denormalize_gripper_actions(actions)
        return {"actions": actions}


def _normalize_gripper_actions(actions: np.ndarray) -> np.ndarray:
    """
    Normalize gripper actions from [-3, 0] to [0, 1].

    Args:
        actions: Action array of shape [..., 14] where the last 2 dims are gripper positions

    Returns:
        Normalized actions
    """
    actions = actions.copy()
    # Gripper indices: 6 (left) and 13 (right)
    # Normalize from [-3, 0] to [0, 1]: (x - min) / (max - min) = (x - (-3)) / (0 - (-3)) = (x + 3) / 3
    actions[..., 6] = (actions[..., 6] + 3.0) / 3.0
    actions[..., 13] = (actions[..., 13] + 3.0) / 3.0
    return actions


def _denormalize_gripper_actions(actions: np.ndarray) -> np.ndarray:
    """
    Denormalize gripper actions from [0, 1] back to [-3, 0].

    Args:
        actions: Normalized action array of shape [..., 14]

    Returns:
        Denormalized actions
    """
    actions = actions.copy()
    # Gripper indices: 6 (left) and 13 (right)
    # Denormalize from [0, 1] to [-3, 0]: x * (max - min) + min = x * 3 - 3
    actions[..., 6] = actions[..., 6] * 3.0 - 3.0
    actions[..., 13] = actions[..., 13] * 3.0 - 3.0
    return actions
