## Environment Prepare
Tips:`vcr`其实是VideoCaptionReconstruction的缩写
,Reconstruction是我要重建这个项目，不是model的名字
```

conda create --name vcr python=3.8.20
conda activate vcr
python -m pip install --upgrade pip
pip install ftfy regex tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install git+https://github.com/openai/CLIP.git

# gpu版本torch
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    
# cpu版本torch
pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu


pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple

#安装cv
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python opencv-contrib-python

#安装transformers
pip install transformers==4.46.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install hf_xet
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -U tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
pip show protobuf
pip install -U protobuf==3.20.3
# 如果你更喜欢 4.x 也行：pip install -U protobuf==4.23.4

# 安装机器学习库 (用于方案0 K-Means聚类)
pip install scikit-learn==1.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Evaluationo Prepare
#### 1、根据README06安装JDK11，为
#### 2、根据README07安装pycocoevlcap和pycocotools等包
#### 3、README07其实有测试步骤，但是这里写了一个`test_eval.py`
可以单独运行
```markdown
python test_eval.py
```
## Dataset Preparation



**For MSRVTT**

直接按照这个直链下载，先把标注文件下载下来
```sh
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
# 它这个MSRVTT_data.json文件里面的split标注默认都是train，所以具体还需要根据额外的csv或者别的方式进行划分
```
这里为了方便工程加载，数据和工程文件分开存放
然后创建一个`datasets/MSRVTT`文件夹,然后按照如下方式把上面那个解压的子文件按照如下方式放置，
然后现在我们去下载原始video的mp4文件
```sh
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
```
然后创建一个`datasets/MSRVTT/raw/`文件夹, 然后去官网链接下载10k的videos全放`raw`里面,
然后创建一个`datasets/MSRVTT/feats/`文件夹,准备用CLIP提取特征
```markdown
datasets/
    ├── MSRVTT/
        ├── raw/#10000videos
        │   ├── video0.mp4
        │   ├── video1.mp4
        │   ├── ...
        ├── feats/
        ├── MSRVTT_data.json
        ├── MSRVTT_JSFUSION_test.json #暂时用不到
        ├── MSRVTT_train.7k.csv #暂时用不到
        ├── MSRVTT_train.9k.csv #暂时用不到
project/
    ├── dataloaders/
    │   ├── ...
```
---
>在`datasets/MSRVTT/`文件夹下创建一个`msrvtt.csv`文件一共10001行
> ```markdown
> video_id
> video0
> ...
> video9999
>```
---





**For MSVD**

标注（captions）下载链接：
```sh
https://www.kaggle.com/datasets/vtrnanh/msvd-dataset-corpus?resource=download
```

视频（YouTubeClips）下载链接：
```sh
https://www.cs.utexas.edu/~ml/clamp/videoDescription/YouTubeClips.tar
```

## Pretrained Model Preparation-Before Feature Extraction
> `models/`默认放在project下面，主要用来存放预训练模型的权重

`extract_clip_feats.py`用来提取clip的最终层的Image或者Text的Final特征,由于本次主要需要VideoFrame的帧的特征，
所以这里主要先做了帧的提取代码，文本的Text代码还没做。
> 目前先做了MSRVTT的特征提取，默认采用的是K-split=12的模式，每个视频提取12个帧的特征，分成12个段，每段提取一个帧，然后进行K-split。
> 每一段采用的是第一帧的位置的帧率提取特征

>在extract_clip_feats.py里面单独运行以下函数，注意这里要魔法然后缓存权重到本地

>第一次借助魔法运行test_load_pretrained_clip()，缓存到本地cache_dir
【这一步可以通过在win上运行，然后把它上传到linux】

>第二次运行test_load_pretrained_clip()直接从本地缓存cache_dir中加载clip权重
```markdown
test_load_pretrained_clip()
```

现在直接在extract_clip_feats.py里面运行`extract_clip_global_vision_feats()`函数
就是把`if __name__ == "__main__":`下面的代码注释掉，然后直接运行`extract_clip_global_vision_feats()`函数

默认采用K_split=12进行采样12帧，保存的格式参考`extract_clip_feats.md`的说明文档

同理，在windows上运行一遍`load_tokenizers.py`，然后上传到linux缓存到同样相对位置的目录下


## Clip Features Extract

直接按默认运行它`extract_clip_feats.py`就能得到一个k_split=12的MSRVTT特征，包括掩码
，注意数据集和HF的风格mask=1的位置表示有效数据，但是transformer的mask=1表示是要忽略的数据


## Training
### Train Base MeanPooling Model
多卡并行
```markdown
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train_base_mean.py   --ddp 1   --gpus "0,1,2,3"   --batch_size 64   --accum_steps 2   --epochs 15   --lr 5e-4

```

## Inference
### Infer Base MeanPooling Model
极速多卡推理
```markdown
python infer_base_mean.py \
  --dataset_type msrvtt \
  --clip_global_vision_feats_path ../datasets/MSRVTT/feats/ViT-B-32_k_split_ks12_features.pickle \
  --annotations_path ../datasets/MSRVTT/MSRVTT_data.json \
  --run_dir /mnt/sda/Disk_D/zhangwei/projects/VideoCaption_Reconstruction/project/runs/base_mean/ \
  --out_dir ./eval/base

```

## Evaluation
### Evaluate Base MeanPooling Model
```markdown

```