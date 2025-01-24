import torch
import json
import datetime
import os

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from functools import partial

from util.vision_util import process_vision_info
from util.logutil import init_logger, get_logger

from accelerate import Accelerator, DeepSpeedPlugin

# Create a DeepSpeedPlugin configuration object to customize DeepSpeed integration settings。
deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=3,   # Enable ZeRO (Zero Redundancy Optimizer) stage 3 optimization
                    # ZeRO stages: 
                    # 0 - disabled
                    # 1 - optimizer state partitioning
                    # 2 - optimizer state + gradient partitioning
                    # 3 - optimizer state + gradient + parameter partitioning (most memory efficient)
    gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps before optimization
    zero3_save_16bit_model=True # Save models in 16-bit precision when using ZeRO stage 3
                                # Reduces model checkpoint size by 50% while maintaining model quality
)

# Initialize the Hugging Face Accelerator with DeepSpeed integration
# Accelerator provides a unified interface for distributed training across various backends
# (TPU, multi-GPU, DeepSpeed, etc.) while maintaining compatibility with PyTorch code
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
'''
Under the above configuration, when launching the script with the command:
CUDA_VISIBLE_DEVICES="0,1" accelerate launch --mixed_precision=bf16 --dynamo_backend=no --num_machines=1 --num_processes=2 --use_deepspeed finetune_distributed.py,
the final DeepSpeed configuration required will be generated during the subsequent execution of accelerator.prepare(). The configuration details are as follows:

json = {
    "train_batch_size": 4, 
    "train_micro_batch_size_per_gpu": 1, 
    "gradient_accumulation_steps": 2, 
    "zero_optimization": {
        "stage": 3, 
        "offload_optimizer": {
            "device": "none", 
            "nvme_path": null
        }, 
        "offload_param": {
            "device": "none", 
            "nvme_path": null
        }, 
        "stage3_gather_16bit_weights_on_model_save": true
    }, 
    "gradient_clipping": 1.0, 
    "steps_per_print": inf, 
    "bf16": {
        "enabled": true
    }, 
    "fp16": {
        "enabled": false
    }, 
    "zero_allow_untested_optimizer": true
}

'''

'''
Attention: 
In DeepSpeed, fp16 and bf16 are generally indicative of mixed precision training. 
The half-precision is used for forward and backward computations, while fp32 is used for optimizer computation.
'''

device = accelerator.device
output_dir = f'train_output/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'

if accelerator.is_local_main_process:
    os.makedirs(output_dir, exist_ok=True)
    init_logger(output_dir)
    logger = get_logger()


class ToyDataSet(Dataset): # for toy demo, for train_data/data.json
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def find_assistant_content_sublist_indexes(l):
    '''
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]}, 
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2) # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))

def collate_fn(batch, processor, device):
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant")
    # [151644, 77091]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>")
    # [151645]
    
    messages = [m['messages'] for m in batch]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list) # -100 is the ignore index in loss function
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    return inputs, labels_ids

def write_chat_template(processor, output_dir):
    '''
    ***Note**

    We should have not had this function, as normal processor.save_pretrained(output_dir) would save chat_template.json file.
    However, on 2024/09/05, I think a commit introduced a bug to "huggingface/transformers", which caused the chat_template.json file not to be saved. 
    See the below commit, src/transformers/processing_utils.py line 393, this commit avoided chat_template.json to be saved.
    https://github.com/huggingface/transformers/commit/43df47d8e78238021a4273746fc469336f948314#diff-6505546ec5a9ab74b2ce6511681dd31194eb91e9fa3ce26282e487a5e61f9356

    To walk around that bug, we need manually save the chat_template.json file.

    I hope this bug will be fixed soon and I can remove this function then.
    '''
    
    output_chat_template_file = os.path.join(output_dir, "chat_template.json")
    chat_template_json_string = json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True) + "\n"
    with open(output_chat_template_file, "w", encoding="utf-8") as writer:
        writer.write(chat_template_json_string)
        logger.info(f"chat template saved in {output_chat_template_file}")

def train():
    # Load the model on the available device(s)
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-2B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )
    
    # ** WARNING ** When run below line , we got below warning message:
    #   Unrecognized keys in `rope_scaling` for 'rope_type'='default': {'mrope_section'}"
    # It is a issue, see https://github.com/huggingface/transformers/issues/33401
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="bfloat16", attn_implementation="flash_attention_2"
    )


    # Load processor. 
    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28

    # **Note:** About padding_side parameter, it default value is "left", here we set it as "right".
    # For why, read below.
    # Typically, in training, when batch size of training dataloader is > 1, it is often we need pad shorter inputs to the same length.
    # To pad, we often add "padding_token_id" to the right side of shorter inputs to make them the same length and set 0 in attention_mask for those padding_token_id.
    # It makes casual_mask easier to build by attention mask. for more detail, see *** notes.txt *** of this repo.
    # BTW, in batching inference, we must use "padding_side" left, as generation usually uses the last token of output list of tokens.
    # 
    # If you like to read more, here are more discussions about padding and padding side:
    # https://github.com/huggingface/transformers/pull/26572
    # https://github.com/pytorch/pytorch/issues/110213
    # transformers/models/qwen2_vl/modeling_qwen2_vl.py: causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=256*28*28, max_pixels=512*28*28, padding_side="right")

    train_loader = DataLoader(
        ToyDataSet("train_data/data.json"),
        batch_size=1,
        collate_fn=partial(collate_fn, processor=processor, device=device)
    )

    model.train()
    epochs = 10
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    for epoch in range(epochs):
        steps = 0
        for batch in train_loader:
            steps += 1
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                inputs, labels = batch
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                if accelerator.is_local_main_process:
                    logger.info(f"Batch {steps} of epoch {epoch + 1}/{epochs}, training loss : {loss.item()}")

    # Synchronize all processes to ensure training completion before saving the model.
    accelerator.wait_for_everyone()
    # Unwrap the model from distributed training wrappers
    unwrapped_model = accelerator.unwrap_model(model)
    # Save the model using HuggingFace's pretrained format
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,# Only save from main process to avoid conflicts
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),# Get complete state dict including optimizer states (critical for DeepSpeed)
    )
    if accelerator.is_local_main_process:
        processor.save_pretrained(output_dir)
        write_chat_template(processor, output_dir)

if __name__ == "__main__":
    train()

