import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parents[1].as_posix()+"/util")  # TODO: fix this ugly import hack


import json
import datetime
import os

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch.optim import AdamW
from logutil import init_logger, get_logger
from util.coco_dataloader import get_train_data_loader
from accelerate import Accelerator

accelerator = Accelerator(gradient_accumulation_steps=2)
device = accelerator.device

if accelerator.is_local_main_process:
    output_dir = f'train_output/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'
    init_logger(output_dir)
    logger = get_logger()

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
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="bfloat16"
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
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=256*28*28, max_pixels=256*28*28, padding_side="right")

    train_batch_size = 4
    train_loader = get_train_data_loader(processor=processor, device=device, batch_size=train_batch_size)

    model.train()
    epochs = 10
    optimizer = AdamW(model.parameters(), lr=1e-5)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    for epoch in range(epochs):
        steps = 0
        avg_loss = 0
        for batch in train_loader:
            steps += 1
            inputs, labels = batch
            if inputs is None and labels is None:
                logger.info(f"Epoch: {epoch + 1}/{epochs}, steps: {steps}/{len(train_loader)}, image size is not (640, 480), skip this batch.")
                continue
            with accelerator.accumulate(model):
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                avg_loss += loss.item()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.is_local_main_process:
                logger.info(f"Epoch: {epoch + 1}/{epochs}, steps: {steps}/{len(train_loader)}, average training loss : {avg_loss / steps}, batch_size:{train_batch_size}, input_ids_shape:{inputs['input_ids'].shape}")

            if steps % 3000 == 0 or steps == len(train_loader):
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    model_checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}-steps-{steps}-loc_size-100-grad_accu-2-640X480-only")
                    os.makedirs(model_checkpoint_dir, exist_ok=True)
                    logger.info(f"Saving model checkpoint to {model_checkpoint_dir}")
                    unwrapped_model.save_pretrained(
                        model_checkpoint_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
                    processor.save_pretrained(model_checkpoint_dir)
                    write_chat_template(processor, model_checkpoint_dir)

if __name__ == "__main__":
    train()