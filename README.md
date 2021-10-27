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
    └── pretrain_ab_basic/
```

To evaluate our pre-trained models, type:

```sh
# evaluate anchor-based model
# cd to src folder and run
python dsnet-py.py
```

You will get a F-score results as follows.

|              | TVSum | SumMe |
| ------------ | ----- | ----- |
| Anchor-based | 62.05 | 50.19 |

