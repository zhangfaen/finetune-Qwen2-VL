import torch
import json
import datetime
import os

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from functools import partial
from util.vision_util import process_vision_info
from torch.optim import AdamW
from util.logutil import init_logger, get_logger

output_dir = f'train_output/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'
init_logger(output_dir)
logger = get_logger()

device = "cuda:0"

class ToyDataSet(Dataset): # for toy demo
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def find_assistant_content_sublist_indexes(l):
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant")
    # [151644, 77091]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>")
    # [151645]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 1):
        # Check if the current and next element form the start sequence
        if l[i] == 151644 and l[i + 1] == 77091:
            start_indexes.append(i)
            # Now look for the first 151645 after the start
            for j in range(i + 2, len(l)):
                if l[j] == 151645:
                    end_indexes.append(j)
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))

def collate_fn(batch, processor, device):
    '''
    A message from data.json may look like below:
        [[{'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}]]
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    The key of this function is to find the indexes of the assistant content in the input_ids list to build labels.
    '''
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
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)

    return inputs, labels_ids

def train():
    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
    )

    # (Pdb++) model
    # Qwen2VLForConditionalGeneration(
    #   (visual): Qwen2VisionTransformerPretrainedModel(
    #     (patch_embed): PatchEmbed(
    #       (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
    #     )
    #     (rotary_pos_emb): VisionRotaryEmbedding()
    #     (blocks): ModuleList(
    #       (0-31): 32 x Qwen2VLVisionBlock(
    #         (norm1): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
    #         (norm2): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
    #         (attn): VisionSdpaAttention(
    #           (qkv): Linear(in_features=1280, out_features=3840, bias=True)
    #           (proj): Linear(in_features=1280, out_features=1280, bias=True)
    #         )
    #         (mlp): VisionMlp(
    #           (fc1): Linear(in_features=1280, out_features=5120, bias=True)
    #           (act): QuickGELUActivation()
    #           (fc2): Linear(in_features=5120, out_features=1280, bias=True)
    #         )
    #       )
    #     )
    #     (merger): PatchMerger(
    #       (ln_q): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
    #       (mlp): Sequential(
    #         (0): Linear(in_features=5120, out_features=5120, bias=True)
    #         (1): GELU(approximate='none')
    #         (2): Linear(in_features=5120, out_features=1536, bias=True)
    #       )
    #     )
    #   )
    #   (model): Qwen2VLModel(
    #     (embed_tokens): Embedding(151936, 1536)
    #     (layers): ModuleList(
    #       (0-27): 28 x Qwen2VLDecoderLayer(
    #         (self_attn): Qwen2VLSdpaAttention(
    #           (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
    #           (k_proj): Linear(in_features=1536, out_features=256, bias=True)
    #           (v_proj): Linear(in_features=1536, out_features=256, bias=True)
    #           (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
    #           (rotary_emb): Qwen2RotaryEmbedding()
    #         )
    #         (mlp): Qwen2MLP(
    #           (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
    #           (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
    #           (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
    #           (act_fn): SiLU()
    #         )
    #         (input_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
    #         (post_attention_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
    #       )
    #     )
    #     (norm): Qwen2RMSNorm((1536,), eps=1e-06)
    #   )
    #   (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
    # )


    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-2B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processor, min image tokens 256, max image tokens 512
    # Note: typically, in training, when we batch size of training dataloader is > 1, it is often we need pad shorter inputs to the same length.
    # in training, we often add "padding_token_id" to the right side of shorter inputs to make them the same length. padding_side right 
    # make casual_mask easier to build by attention mask. for more detail, see *** notes.txt *** of this repo.
    # in batching inference, we must use "padding_side" left, as generation usually ust last token of output list of tokens.
    # in training, it is recommended to use "padding_side" right.
    # https://github.com/huggingface/transformers/pull/26572 
    
    # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
    # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
    # Details: https://github.com/pytorch/pytorch/issues/110213
    # see transformers/models/qwen2_vl/modeling_qwen2_vl.py: causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=256*28*28, max_pixels=512*28*28, padding_side="right")

    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    train_loader = DataLoader(
        ToyDataSet("train_data/data.json"),
        batch_size=2,
        collate_fn=partial(collate_fn, processor=processor, device=device)
    )

    model.train()
    epochs = 300
    # import pdb
    # pdb.set_trace()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    NUM_ACCUMULATION_STEPS = 2
    for epoch in range(epochs):
        accumulated_avg_loss = 0
        steps = 0
        for batch in train_loader:
            steps += 1
            inputs, labels = batch
            # import pdb
            # pdb.set_trace()
            outputs = model(**inputs, labels=labels)
            
            loss = outputs.loss / NUM_ACCUMULATION_STEPS
            accumulated_avg_loss += loss.item()
            loss.backward()
            
            if steps % NUM_ACCUMULATION_STEPS == 0:
                logger.info(f"Batch {steps} of epoch {epoch + 1}/{epochs}, average training loss of previous {NUM_ACCUMULATION_STEPS} batches: {accumulated_avg_loss}")
                accumulated_avg_loss = 0
                optimizer.step()
                optimizer.zero_grad()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    train()

    
