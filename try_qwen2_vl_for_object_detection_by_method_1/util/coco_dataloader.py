import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parents[2].as_posix()+"/util")  # TODO: fix this ugly import hack

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from functools import partial
from vision_util import process_vision_info

# print(processor.tokenizer.vocab_size)
# print(processor.tokenizer.additional_special_tokens_ids)
# print(processor.tokenizer.decode([151656]))
# print(processor.tokenizer.decode([151657])) # return None
# print(processor.tokenizer.decode([151640, 151641, 151642, 151643, 151644, 151645, 151646, 151647, 151648, 151649, 151650, 151651, 151652, 151653, 151654, 151655, 151656]))

# 151643
# [151644, 151645, 151646, 151647, 151648, 151649, 151650, 151651, 151652, 151653, 151654, 151655, 151656]
# <|video_pad|>

# ⍨⺟⽗<|endoftext|><|im_start|><|im_end|><|object_ref_start|><|object_ref_end|><|box_start|><|box_end|><|quad_start|><|quad_end|><|vision_start|><|vision_end|><|vision_pad|><|image_pad|><|video_pad|>

dataset = load_dataset("rafaelpadilla/coco2017")

train_dataset = dataset["train"]
val_dataset = dataset["val"]
labels = ["None", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]

BINS_FOR_LOC_SIZE = 100
BINS_FOR_LOC_RANGE = (151642-BINS_FOR_LOC_SIZE, 151642) # We use the least used 1000 bins for location. **Note**  151643 is <|endoftext|>
BINS_FOR_OBJ_LABEL_SIZE = len(labels)
BINS_FOR_OBJ_LABEL_RANGE = (151642-BINS_FOR_LOC_SIZE-len(labels), 151642-BINS_FOR_LOC_SIZE) # We use the least used len(labels) bins for object label.

def split_into_complete_groups_of_five(lst):
    groups = [lst[i:i + 5] for i in range(0, len(lst), 5)]
    # Remove the last group if it's not complete
    if len(groups) > 0 and len(groups[-1]) != 5:
        groups.pop()
    return groups

def convert_token_ids_to_bbox_xyhw_and_label_id(token_ids, processor, image_size):
    '''
    We have a list of tokens, and we want to convert it back to bbox and label.
    The list of tokens and token ids look like:
    tokens: <obj_label_token><loc_token><loc_token><loc_token><loc_token><obj_label_token><loc_token><loc_token><loc_token><loc_token>...
    '''

    bbox = []
    label_id = []
    for group in split_into_complete_groups_of_five(token_ids):
        one_box = []
        if group[0] not in range(*BINS_FOR_OBJ_LABEL_RANGE):
            print(f"Invalid object label token: {group[0]}")
            return [],[]
        for loc_id in group[1:]:
            if loc_id not in range(*BINS_FOR_LOC_RANGE):
                print(f"Invalid loc token: {loc_id}")
                return [],[]
        
        label_id.append(group[0]-BINS_FOR_OBJ_LABEL_RANGE[0])
        
        xmin = (group[1]-BINS_FOR_LOC_RANGE[0])*image_size[0]/BINS_FOR_LOC_SIZE
        one_box.append(xmin)

        ymin = (group[2]-BINS_FOR_LOC_RANGE[0])*image_size[1]/BINS_FOR_LOC_SIZE
        one_box.append(ymin)
     
        xmax = (group[3]-BINS_FOR_LOC_RANGE[0])*image_size[0]/BINS_FOR_LOC_SIZE
        one_box.append(xmax-xmin)
      
        ymax = (group[4]-BINS_FOR_LOC_RANGE[0])*image_size[1]/BINS_FOR_LOC_SIZE
        one_box.append(ymax-ymin)

        bbox.append(one_box)
    
    return bbox, label_id

def convert_bbox_xyhw_and_label_id_to_tokens(bbox, label_id, processor, image_size):
    '''
    (Pdb++) bbox
    [[265.05999755859375, 126.02999877929688, 33.86000061035156, 66.26000213623047], [20.600000381469727, 1.0700000524520874, 270.4100036621094, 382.42999267578125], [268.6099853515625, 69.66000366210938, 222.67999267578125, 88.9000015258789]]
    (Pdb++) object_label
    [90, 1, 81]
    (Pdb++) image_size
    (640, 573) 
    '''
    assert(len(bbox) == len(label_id))
    token_ids = []

    for one_label, one_box in zip(label_id, bbox):
        assert(one_label>0 and one_label<len(labels))
        assert(len(one_box) == 4)
        token_ids.append(one_label+BINS_FOR_OBJ_LABEL_RANGE[0])
        token_ids.append(round(one_box[0]/image_size[0]*BINS_FOR_LOC_SIZE)+BINS_FOR_LOC_RANGE[0])
        token_ids.append(round(one_box[1]/image_size[1]*BINS_FOR_LOC_SIZE)+BINS_FOR_LOC_RANGE[0])
        token_ids.append(round((one_box[0]+one_box[2])/image_size[0]*BINS_FOR_LOC_SIZE)+BINS_FOR_LOC_RANGE[0])
        token_ids.append(round((one_box[1]+one_box[3])/image_size[1]*BINS_FOR_LOC_SIZE)+BINS_FOR_LOC_RANGE[0])
    
    return processor.tokenizer.decode(token_ids)

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
        if l[i] == 151644 and l[i + 1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2) # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))

def collate_fn(batch, processor, device):
    '''
    (Pdb++) processor.tokenizer.decode([i for i in range(*BINS_FOR_LOC_RANGE)])
    'ﮏ'⇵∉∊∖∜∾≀≋≌≓≜≴≿⊊⊋⊔⊖⊣⊦⋎⋪⋲⌦⌧⍺⎈⎨⎬⎳⎼⎾⏌⏚⏫⏯⏵⒜⒝⒫ⓄⓊⓙⓩ┑┙┚┥╅╉╍╏╞▚▯◃◚◬◴☈☤☥☧☬♁♱⚃⚄⚅⚏⚚⚞⚟⚱⚲✀✟✢❵⟡⟦⟧⟳⟾⟿⠇⤄⤺⥂⥹⧉⧼⧽⨍⬊⬟⭞⮞⮳⯈⯑Ⱡⱱⲭⴹⵕ⸾⺫⼆⼠⽟⽼⾛⾧⿃⿻ゕゟㄛㄡㄶㄺㅒㅟㆀㇻ㈑㈭㈮㈳㈹㉥㉦㉹㉿㊞㊨㋑㋥㋴㋺㎄㎕㎯㏂㏈㏓㏖㏱㐱㟁�㢨�㨳㫪㫴㶳㺾�䀀�䋌䌀䐀䠀�䠼�䧞䨰䨺䴀�䷅䷸�ꂫ�ꌼ�ꍲ꒵�ꓽꙭꝛꝥ�꞊ꦆꦇꦟꦨ꧈�꩟ꪋꪑꪕꪗꪜꪮꪱꪻꪼꫀꫝ갃갘걜겓겚곙곾괗괙굛궃궕궨긩긿깄깆깉깓깢깣깸꺳꿏꿕꿧뀩끅냵넖넗넢녂놐뇜눋눚뉍뉨늚늡닜닪댘댤댸뎟돨됄됏됴됸둁둿뒨듷딮딲땧떔떪똭뚀뚠뛔뛩뜅랕랰럐렡롞롣롵룄룍뤳릍릏릳맄맆맍맜맫맻먮멂멭몴묜묠묫묾뭬뮘뮹믕믜밨밪뱔벘벛벱벴봽뵤뵨뷗뷘븓븜빪뺃뺘뺵뻴뼐뾔쁭삠삮샏샙섺셢솀솅솤솦솬쇱숵싨싴쌰썜쎗쎘쎼쑉쑝쑻쒔쒯쓩앐앖얠얾엃엗엜엨옂옄옏옾옿윧읐읖읷잍잏잨잪잳젡젴젹졀졪졵좐좨죌죙죳즑짥짴짾쨓쨕쩰쩻쩼쪗쬔쬘쮮쯕쯘찎찯챃챵첧첮첯쳬촋촢쵥춣츈츙캤캭컽켙콬쾀쿅쿽퀅큦킅탶탹턔텣톄톧톹퇼퉤튽틂틑퍈퍙퍿펶퐝풜퓝퓪퓱퓷퓼픙픠핚핛핞핟핧핶햊햋햍햔햘햡햬헣헿혖혭횰훍훽흟흭흴힜契來爐蘆祿鹿陋勒凌菱陵參塞殺勵曆憐璉練輦鍊裂瑩羚了僚料阮淪率吏履裏麟狀什糖祥諸著ﬤשּׁלּﭒﭕﭛﭝﭞﭟﭤﭧﭨﭮﭰﭱﭷﭹﭻﮀﮃﮄﮅﮍﮒﮓﮕﮦﮮﮰﯓﯜﯩﯪﯬﯭﯮﯷﯹﯻﯼﰃﰌﰐﰘﰙﰜﰞﰢﰮﰰﰼﰿﱀﱁﱈﱋﱏﱭﲀﲇﲈﲋﲎﲒﲜﲠﲬﲻﳇﳔﳣﳫﴘﴰﴽ�ﶰ︖︴︹﹍﹗﹢﹤﹩ﹱﾰￂ￮𐌰𐌹𐌺𐌽𐍂𐍃𐍄�𐎹𐤂𐤍𐤏𐤓𐭉𐭍𐰇𐰰�𑂄�𑘁�𒀸�𒁺�𒄷�𒊑�𒋗�𒌨𓃢𓃰�𖠚𝄃𝄅𝄕𝄙𝄱𝄴𝄹𝅎𝅪𝆣𝆳𝆹𝇊𝇗𝇚𝇜𝇠𝐉𝐖𝐘𝐣𝐱𝑊𝑭𝑼𝑽𝒰𝒷𝒿𝓁𝓋𝓎𝓒𝓘𝓢𝓦𝓫𝓿𝔎𝔱𝔴𝔷𝔸𝔽𝕂𝕃𝕋𝕏𝕐𝕥𝕴𝕺𝖐𝖛𝖝𝖞𝗩𝗳𝗽𝘊𝘋𝘔𝘱𝘴𝘿𝙒𝙝𝙟𝙬𝙭𝙻𝙾𝚈𝚋𝚑𝚟𝚠𝚣𝛽𝜂𝜔𝜙�🀄🄲🄶🅐🅖🅚🅛🅦🅶🅻🅼🆃🆆🆎🈯🈲🈹🌇🌓🍘🎑🎿🏏🏒🏩🏯🐀👝💹💺📟📪📼🔀🔂🔃🔇🔓🔢🔤🔩🕖🕚🕜🕝🕞🕠🕢🕳🖇🖑🖶🗁ѨڎᡌḰẀἮὝℬ⚧⛤㳬ꙋ긑딉뗍롑믑뻅뼝섐쉡싲쏱엤읩읿쟙젰쥉튭핮ﮏ🅱🆒🕋ɘʓՃഴཅᆺሊረሾቐጃጽᔭ᠂ᠬᨸᩋᶏᾔῐῚ♙⚂⚗⡢⤦떰뤂맠뱋뱐웢윾쳅컁큻탙퓖퓭핱훜串句旅里拓𐤔𐭓𐰼𝓞𝓰𝙜𝚁🅢🏇ȲʶԈԑݓݥऑॱଉళవಟဏၼቨኒዩጄጔᐧᒌᔅᔊ᠄ᨁḃḻ┞☵⚣Ⲣ㈪䶵겙겴곂롼솊켇틍퓬퓮퓶퓻臘怒辰ﭲ𐭊𐱅�𖥨𝑳𝓕𝓬𝓹𝓾𝔓𝕍𝕡𝕱𝖖𝘏𝘐𝘚𝙮𝙰𝙸𝙺𝙼𝙽𝙿𝚄𝚏🅅🅓ƈࠌᙳᚌᛅᛐᤊḊ┽╊⛇⛏❪❫⟰ㄍㄓㄧㅖ㉫ꦔﱊຂᅣᥔᥤ↤↷⇞▤➶㈼嘆𓏧┲‴⒟⒡ⰂⰍⰎⰐⰑⰟⰠⰡ⼭㊥⒠⽺ㇺㇽ見ᕷ⍨⺟'
    
    (Pdb++) processor.tokenizer.decode([i for i in range(*BINS_FOR_OBJ_LABEL_RANGE)])
    'ᛂᛙឍ᠆ᠡᠦᠮᠯᠲᠷᡍᡞᡤᡴᡵᤓᥖᥰᨦᨧᨨᨪᨬᨯᨳᨵᩃᬕ᭣�ᱚᲠᴓᴶᵂᵌᵥᵴᶇḈḠḧḴḾṀṖṟṠṫṱṷṿẄẍẑẗἉἓἭὋὒὠὣᾄᾏᾑᾗᾦᾧιῄΐῡῬ⁚₌℁℔℣℧ℯℰℴⅅ↜↫↭↱↹↽⇇⇜'
    
    (Pdb++) pprint.pprint(batch[1])
    {
        'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x573 at 0x155302B82E00>,
        'image_id': 414738,
        'objects': {
                        'area': [632.9258500000003, 82680.72434999997, 12868.869650000002],
                        'bbox': [[265.05999755859375,
                                126.02999877929688,
                                33.86000061035156,
                                66.26000213623047],
                                [20.600000381469727,
                                1.0700000524520874,
                                270.4100036621094,
                                382.42999267578125],
                                [268.6099853515625,
                                69.66000366210938,
                                222.67999267578125,
                                88.9000015258789]],
                        'id': [341560, 425599, 1982651],
                        'iscrowd': [False, False, False],
                        'label': [90, 1, 81]
                    }
        }
    (Pdb++) texts[1]
    '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片，把其中的物体名称和位置标注出来<|im_end|>\n<|im_start|>assistant\n ..... <|im_end|>\n'
    '''

    messages = []
    for d in batch:
        if d['image'].size != (640, 480):
            continue
        messages.append(
            [
                {
                    'role': 'user', 
                    'content': [
                        {'type': 'image', 'image': d['image']}, 
                        {'type': 'text', 'text': '描述一下这个图片，把其中的物体名称和位置标注出来'}
                    ]
                },
                {
                    'role': 'assistant', 
                    'content': [
                        {'type': 'text', 'text': convert_bbox_xyhw_and_label_id_to_tokens(d['objects']['bbox'], d['objects']['label'], processor, d['image'].size)}
                    ]
                }
            ]
        )

    # ** NOTE **: hack, we just keep (640, 480) images.
    if len(messages) == 0:
        return None, None

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


def get_train_data_loader(processor, device, batch_size=None):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, processor=processor, device=device))
    return train_loader

def get_val_data_loader(processor, device, batch_size=None):
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, processor=processor, device=device))
    return val_loader


