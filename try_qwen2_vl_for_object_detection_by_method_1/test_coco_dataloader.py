
from transformers import AutoProcessor
from util.coco_dataloader import get_train_data_loader, convert_bbox_xyhw_and_label_id_to_tokens, convert_tokens_to_bbox_xyhw_and_label_id, dataset

if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=256*28*28, max_pixels=512*28*28, padding_side="right")

    bbox=[[265.05999755859375, 126.02999877929688, 33.86000061035156, 66.26000213623047], [20.600000381469727, 1.0700000524520874, 270.4100036621094, 382.42999267578125], [268.6099853515625, 69.66000366210938, 222.67999267578125, 88.9000015258789]]
    label_id=[90, 1, 81]
    image_size= (640, 573) 



    print(f"bbox: {bbox}, \nlabel_id: {label_id}")
    tokens = convert_bbox_xyhw_and_label_id_to_tokens(bbox, label_id, processor, image_size)
    print(f"tokens:{tokens}")
    print(f"token_ids:{processor.tokenizer.encode(tokens)}")
    # (Pdb++) processor.tokenizer.encode(tokens)
    # [150640, 151056, 150862, 151109, 150978, 150551, 150674, 150644, 151097, 151311, 150631, 151062, 150764, 151410, 150919]

    bbox, label_id = convert_tokens_to_bbox_xyhw_and_label_id(tokens, processor, image_size)
    # (Pdb++) bbox
    # [[264.96, 126.06, 33.920000000000016, 66.46799999999999], [20.48, 1.146, 270.71999999999997, 382.191], [268.8, 69.906, 222.71999999999997, 88.815]]
    # (Pdb++) label_id
    # [90, 1, 81]
    # (Pdb++) 
    print(f"bbox: {bbox}, \nlabel_id: {label_id}")

    # for batch in get_train_data_loader(processor, device='cpu', batch_size=1):
    #     import pdb
    #     pdb.set_trace()
    #     print(batch)
    counter = {}
    for d in dataset["train"]:
        if d['image'].size in counter:
            counter[d['image'].size] += 1
        else:
            counter[d['image'].size] = 1
    print(counter)
        
        