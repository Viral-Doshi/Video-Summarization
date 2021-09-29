# Video-Summarization
## Getting Started

This project is developed on Windows with CUDA 11.1.

First, clone this project to your local environment.

```sh
git clone https://github.com/Viral-Doshi/Video-Summarization.git
```

Create a virtual environment with python 3.6, preferably using Anaconda.

```sh
conda create --name dsnet python=3.6
conda activate dsnet
```

Install python dependencies.

```sh
pip install -r requirements.txt
```


Install cuda enabled pytorch.
For other versions of cuda, select command from [here](https://pytorch.org/).

```sh
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

## Datasets

Download the pre-processed datasets into `datasets/` folder.

```sh
mkdir datasets/
cd datasets/
wget https://www.dropbox.com/s/tdknvkpz1jp6iuz/dsnet_datasets.zip
unzip dsnet_datasets.zip
```
Download and extraction can be done manually.
Download [link](https://www.dropbox.com/s/tdknvkpz1jp6iuz/dsnet_datasets.zip).


Now the datasets structure should look like

```
Video-Summarization
└── datasets/
    ├── eccv16_dataset_ovp_google_pool5.h5
    ├── eccv16_dataset_summe_google_pool5.h5
    ├── eccv16_dataset_tvsum_google_pool5.h5
    ├── eccv16_dataset_youtube_google_pool5.h5
    └── readme.txt
```

## Pre-trained Models

Pre-trained model can also be downloaded.

```sh
mkdir models
cd models
wget https://www.dropbox.com/s/0jwn4c1ccjjysrz/pretrain_ab_basic.zip
unzip pretrain_ab_basic.zip
```
Download [link](https://www.dropbox.com/s/0jwn4c1ccjjysrz/pretrain_ab_basic.zip).
It can be done manually as well.

```
Video-Summarization
└── models/
    └── pretrain_ab_basic
```

To evaluate our pre-trained models, type:

```sh
# evaluate anchor-based model
python evaluate.py anchor-based --model-dir ../models/pretrain_ab_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml
# evaluate anchor-free model
python evaluate.py anchor-free --model-dir ../models/pretrain_af_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4
```

You will get a F-score results as follows.

|              | TVSum | SumMe |
| ------------ | ----- | ----- |
| Anchor-based | 62.05 | 50.19 |

## Training

### Anchor-based

To train anchor-based attention model on TVSum and SumMe datasets with canonical settings, run

```sh
python train.py anchor-based --model-dir ../models/ab_basic --splits ../splits/tvsum.yml ../splits/summe.yml
```

To train on augmented and transfer datasets, run

```sh
python train.py anchor-based --model-dir ../models/ab_tvsum_aug/ --splits ../splits/tvsum_aug.yml
python train.py anchor-based --model-dir ../models/ab_summe_aug/ --splits ../splits/summe_aug.yml
python train.py anchor-based --model-dir ../models/ab_tvsum_trans/ --splits ../splits/tvsum_trans.yml
python train.py anchor-based --model-dir ../models/ab_summe_trans/ --splits ../splits/summe_trans.yml
```

To train with LSTM, Bi-LSTM or GCN feature extractor, specify the `--base-model` argument as `lstm`, `bilstm`, or `gcn`. For example,

```sh
python train.py anchor-based --model-dir ../models/ab_basic --splits ../splits/tvsum.yml ../splits/summe.yml --base-model lstm
```

### Anchor-free

Much similar to anchor-based models, to train on canonical TVSum and SumMe, run

```sh
python train.py anchor-free --model-dir ../models/af_basic --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4
```

Note that NMS threshold is set to 0.4 for anchor-free models.

## Evaluation

To evaluate your anchor-based models, run

```sh
python evaluate.py anchor-based --model-dir ../models/ab_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml
```

For anchor-free models, remember to specify NMS threshold as 0.4.

```sh
python evaluate.py anchor-free --model-dir ../models/af_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4
```

## Generating Shots with KTS

Based on the public datasets provided by [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce), we apply [KTS](https://github.com/pathak22/videoseg/tree/master/lib/kts) algorithm to generate video shots for OVP and YouTube datasets. Note that the pre-processed datasets already contain these video shots. To re-generate video shots, run

```sh
python make_shots.py --dataset ../datasets/eccv16_dataset_ovp_google_pool5.h5
python make_shots.py --dataset ../datasets/eccv16_dataset_youtube_google_pool5.h5
```

## Using Custom Videos

### Training & Validation

We provide scripts to pre-process custom video data, like the raw videos in `custom_data` folder.

First, create an h5 dataset. Here `--video-dir` contains several MP4 videos, and `--label-dir` contains ground truth user summaries for each video. The user summary of a video is a UxN binary matrix, where U denotes the number of annotators and N denotes the number of frames in the original video.

```sh
python make_dataset.py --video-dir ../custom_data/videos --label-dir ../custom_data/labels \
  --save-path ../custom_data/custom_dataset.h5 --sample-rate 15
```

Then split the dataset into training and validation sets and generate a split file to index them.

```sh
python make_split.py --dataset ../custom_data/custom_dataset.h5 \
  --train-ratio 0.67 --save-path ../custom_data/custom.yml
```

Now you may train on your custom videos using the split file.

```sh
python train.py anchor-based --model-dir ../models/custom --splits ../custom_data/custom.yml
python evaluate.py anchor-based --model-dir ../models/custom --splits ../custom_data/custom.yml
```

### Inference

To predict the summary of a raw video, use `infer.py`. For example, run

```sh
python infer.py anchor-based --ckpt-path ../models/custom/checkpoint/custom.yml.0.pt \
  --source ../custom_data/videos/EE-bNr36nyA.mp4 --save-path ./output.mp4
```
