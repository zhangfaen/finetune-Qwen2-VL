# Fine-tuning Qwen2-VL-2B

**[English version of this document](README.en.md)**

### News/Updates
* 2024/09/13 Add a video data example in train_data and test_data.  **Note:** 
  * This need "pip install av" to process video data. See updated requirements.txt file. 
  * For training, video data consumes much GPU ram, if your GPU ram is not enough, you may set the batch size to 1. 
  * To further save GPU ram, you can downgrade kinds of configs for video in util/vision_util.py. 
* 2024/09/12 Support flash_attention2 in training scripts and testing scripts.
  * For detail, see https://github.com/zhangfaen/finetune-Qwen2-VL/issues/2#issuecomment-2345795557 

### Introduction
2024年8月29日，开源大模型界的明星 通义千问团队 发布了多模态大模型Qwen2-VL，有2B、7B和72B 3个版本。整体来看，Qwen2-VL 72B 规模的模型在大部分的指标上都达到了最优，甚至超过了 GPT-4o 和 Claude3.5-Sonnet 等闭源模型，特别是在文档理解方面优势明显。他们在他们的blog上展示了很多炸裂的案例，具体参考 https://qwenlm.github.io/zh/blog/qwen2-vl/ 。 Qwen2-VL模型架构图，画的非常好，我单独引用放在下面。  

<img src="readme_imgs/1.jpeg" width="100%" height="40%">   
    

Qwen2-VL 很棒，相信很多人想在其上继续开发或者改造适配为VLA（Vision Language Action）模型（没错，改造为控制机器人的大模型），但官方在我写这个repo的时候，还没有给出一份简洁的微调或者再训练的代码。  

注：官访给了一个使用LLaMA-Factory方式微调Qwen2-VL的方法，我尝试了一下，发现LLaMA-Factory过于黑盒，概念很多，甚至有点冗余。我自己喜欢简单透明的代码，所以自己写了一份微调Qwen2-VL的代码，希望对喜欢自己写train-loop的同学有帮助。   

我有一个微信订阅号 “后向传播”， 在里面时不时写一些技术类文章，包括这篇 （ https://mp.weixin.qq.com/s/mN9Pxpd2Wciw1-IAoFc08A ），欢迎关注。
<img src="readme_imgs/3.jpg" width="20%" height="20%"> 

### 快速开始微调或者再训练Qwen2-VL 2B模型
---

```bash
%git clone https://github.com/zhangfaen/finetune-Qwen2-VL
%cd finetune-Qwen2-VL
%conda create --name qwen2-VL python=3.10
%conda activate qwen2-VL
%pip install -r requirements.txt
```

我提供了2个微调脚本，一个用于单GPU训练，一个用于多GPU训练。本repo的目的是帮助大家快速上手微调Qwen2-VL，所以只准备了一个非常玩具的数据，见 train_data/data.json文件。同时，训练代码也没有做什么evaluation，只是简单的打印了training loss。如果大家想在训练的过程加入evaluation步骤，可以参考train_data/data.json文件，准备数据，然后修改finetune_distributed.py 或者 finetune.py。   

如果想用单GPU卡微调模型，可以运行如下命令：
```bash
./finetune.sh # 注意这个文件中的CUDA_VISIBLE_DEVICES变量设置为合适的值
```   
如果想用多GPU卡微调模型，可以运行如下命令：
```bash
./finetune_distributed.sh # 注意这个文件中的CUDA_VISIBLE_DEVICES变量设置为合适的值
```   

### 测试微调或者再训练后的模型
---
```bash
%python test_on_official_model.py # 测试官方的Qwen2-VL 2B模型
%python test_on_trained_model_by_us.py # 测试我们微调或者再训练后的Qwen2-VL 2B模型
```   
注：test_on_trained_model_by_us.py这个文件中定义了model_dir，如果你微调了多个模型，可以修改这个文件，指定你微调或者再训练后的模型路径。   

另外，如果你没有足够的GPU（GPU Poor）尝试本repo的代码，你可以下载我微调的一个模型：

```bash
%huggingface-cli download zhangfaen/Qwen2-VL-2B-checkpoint  --local-dir model_checkpoint/
%python test_on_trained_model_by_us.py # 注意修改本文件中的 model_dir 变量值为 "model_checkpoint/"
```

上面test_on_official_model.py 和 test_on_trained_model_by_us.py都是对test_data/下的两个图片做描述。
无论是官方的Qwen2-VL 2B模型，还是我们微调或者再训练后的Qwen2-VL 2B模型，都可以很好的描述这两个图片，但是描述结果稍有不同。

例如，下面这张图：  
<img src="test_data/4.png" width="70%" height="70%">   

官方的Qwen2-VL 2B模型描述结果如下：
```
这张图片中，一个人穿着灰色西装，打着蓝色领带，站在一个带有“KING”标志的背景前。背景中的标志上有中文和英文文字，显示了“KING”品牌或公司名称。这个人看起来很正式，可能是在一个商业或办公环境中。
```

我们微调或者再训练后的Qwen2-VL 2B模型描述结果如下：
```
这张图片展示了一位穿着正式西装的男子。他穿着灰色的西装外套和蓝色的领带，双手交叉放在胸前，显得非常自信和专业。背景是一面白色的墙壁，墙上有一个红色和黑色的标志，上面写着“KING”。整体氛围显得非常正式和专业。
```

另外一张图：  
<img src="test_data/5.png" width="70%" height="70%">   

官方的Qwen2-VL 2B模型描述结果如下：
```
这张图片展示了一个中文搜索引擎的界面，具体是“有道”搜索引擎。页面顶部显示了搜索框和搜索提示，输入框中输入了“beach”，并选择了“英中”选项。搜索结果列出了与“beach”相关的词汇和短语，包括：  
 1. beach - 海滩, 海滨; v. 把（船）拖上岸；搁浅；（垂钓者）钓（鱼）。   
 2. beaches - 海滩, 海滨（beach 的复数）   
 3. beachhead - n. [军] 滩
 ```
我们微调或者再训练后的Qwen2-VL 2B模型描述结果如下：
```
这张图片展示了一个中文搜索引擎的界面，具体是“有道”（youdao）的搜索结果页面。页面顶部显示了“有道”（youdao）的标志和英文名称。页面中间有一个蓝色的矩形框，里面显示了“beach”这个词的翻译和相关词汇。  
在蓝色框内，有三个相关的词汇：beach、beaches和beachhead。每个单词都有对应的英文解释和用法说明。例如，“beach”表示海滩、海滨；“beaches”表示海滩、海滨（beach的复数形式）；  
 ```

总体上，微调后的模型，看起来性能没有受到很大的影响。下面这张图是微调过程中的一个日志文件
<img src="readme_imgs/2.png" width="100%" height="70%">   
可以看到，training loss在下降，说明模型在训练过程中收敛了。


### Acknowledgement
---
This repo is built based on 
- https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/
- https://huggingface.co/zhangfaen/Qwen2-VL-2B-checkpoint/    

Many thanks to them for the great model/data/code!

