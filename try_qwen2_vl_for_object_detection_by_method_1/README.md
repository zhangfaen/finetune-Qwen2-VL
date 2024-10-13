# Try to finetune Qwen2-VL for object detection
Qwen2-VL is a strong VLM.  You may heard that: Florence 2, released by Microsoft in June 2024, is a foundation vision-language model. This model is very attractive because of its small size (0.2B and 0.7B) and strong performance on a variety of computer vision and vision-language tasks. https://github.com/zhangfaen/finetune-Florence-2-large-ft  
![alt text](image.png)  

It could be used for object detection. Qwen2-VL has some support for object detection, but performance is limited (see https://github.com/QwenLM/Qwen2-VL/issues/9). So I try to finetune Qwen2-VL for object detection by my way for fun.

**Note**  This folder has immature code, it is just for fun. I will update it when I have time.

## 1. Prepare data
I use the COCO dataset for object detection. I use the COCO 2017 train and validation dataset. I use the following code to prepare the data.

```python
from datasets import load_dataset
dataset = load_dataset("rafaelpadilla/coco2017")
```

## 2. Prepare environment
```bash
%git clone https://github.com/zhangfaen/finetune-Qwen2-VL
%cd finetune-Qwen2-VL
%conda create --name qwen2-VL python=3.10
%conda activate qwen2-VL
%pip install -r requirements.txt
```