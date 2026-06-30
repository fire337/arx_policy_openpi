from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

# 加载预训练的 PaliGemma
hf_model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-224")
hf_model.to('cuda')
processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")

# 测试
from PIL import Image
import requests

image = Image.open("/workspace/openpi/data/droid_example/dataset-card.jpg")
inputs = processor(images=image, text="describe this image", return_tensors="pt").to("cuda")

# 改善生成质量
outputs = hf_model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
)
print(processor.decode(outputs[0], skip_special_tokens=True))
