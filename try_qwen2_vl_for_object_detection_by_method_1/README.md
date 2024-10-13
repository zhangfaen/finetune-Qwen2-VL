# Try to finetune Qwen2-VL for object detection
Qwen2-VL is a strong VLM.  You may heard that: Florence 2, released by Microsoft in June 2024, is a foundation vision-language model. This model is very attractive because of its small size (0.2B and 0.7B) and strong performance on a variety of computer vision and vision-language tasks. https://github.com/zhangfaen/finetune-Florence-2-large-ft  
![alt text](image.png)  

It could be used for object detection. Qwen2-VL has some support for object detection, but performance is limited (see https://github.com/QwenLM/Qwen2-VL/issues/9). So I try to finetune Qwen2-VL for object detection by my way for fun.

**Note**  
1. This folder has immature code, it is just for fun. I will update it when I have time.
2. The key idea is to re-reuse the least used tokens in the vocab to represent coco data categories (or object label in term of coco) and bounding box coordinates.


## 1. Prepare data
I use the COCO dataset for object detection. I use the COCO 2017 train and validation dataset. I use the following code to prepare the data.

```python
from datasets import load_dataset
dataset = load_dataset("rafaelpadilla/coco2017")
```

## 2. Prepare environment
```bash
%git clone https://github.com/zhangfaen/finetune-Qwen2-VL
%cd finetune-Qwen2-VL/try_qwen2_vl_for_object_detection_by_method_1
%conda create --name qwen2-VL python=3.10
%conda activate qwen2-VL
%pip install -r requirements.txt
%cd ~/temp
%git clone https://github.com/huggingface/transformers
%cd transformers
# Cherry-pick this PR https://github.com/huggingface/transformers/pull/33487 to fix the bug in the qwen2-vl code
%pip install -e .
%cd  finetune-Qwen2-VL/try_qwen2_vl_for_object_detection_by_method_1
```

## 3. Finetune Qwen2-VL for object detection
```bash
%./finetune_coco_distributed.sh
```

## 4. Test the finetuned model
Open test_on_trained_model_by_us.ipynb and play with it.