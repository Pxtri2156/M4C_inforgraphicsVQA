This guideline will help you run M4C model on [infographicVQA](https://arxiv.org/abs/2104.12756) dataset.
# Create enviroment
Firstly, you need to have the enviroment to work.
## Create docker container
I will create docker container to setup the enviroment. You can reference [the docker document](https://docs.docker.com/) to know more for docker. However, M4C model requires a lot of memory. So, you should set large memory for docker container.  
Example: 
```
docker run --ipc=host -it --gpus all --runtime=nvidia --name mlcv_pxtri_docvqa_mmf_v1 \
   --cpus=30 --shm-size=512m  \
   -m 30000M --memory-swap -1 \
   -v /mlcv:/mlcv -v /home:/home \
   nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 /bin/bash
```
## Use container docker 
### start
```
docker start mlcv_pxtri_docvqa_mmf_v1
```
### exec
```
docker exec -it mlcv_pxtri_docvqa_mmf_v1 /bin/bash;
```
## Create environtment in conda 
You need to have conda to build enviroment. After you install conda, you will build your environments. You can refer my requirements file at [here](https://github.com/Pxtri2156/M4C_inforgraphicsVQA/requirements_v2.txt)
Example 
## Activate conda environment 
Example
```
conda create --name mmf_docVQA
pip install -r requirements.txt
```
Then, you can activate enviroment to run code.  
Example
```
conda activate mmf_docVQA
```
# Test code 
Before you try to run anything you need know some environment variables below:  
* `data_dir` where obtain data that include anotation, vocabulary, extracted features.
* `log_dir` where save log of running.
* `save_dir` where save trained models and config files.
* `cache_dir` where save **FastText** model.
* `report_dir` where save predicted results.
* `tensorboard_logdir` where save special logs file to visualize logs with Tensorflow.

Then you can try to run code
```
CUDA_VISIBLE_DEVICES=0 mmf_run config=projects/m4c/configs/textvqa/defaults.yaml \
    datasets=textvqa \
    model=m4c \
    env.save_dir= <path> \
    env.data_dir= <path> \
    env.cache_dir= <path> \
    env.log_dir= <path> \
    env.tensorboard_logdir=<path>\
    env.report_dir=<path>\
    run_type=train_val \
    training.batch_size=8 \
    training.num_workers=1
``` 
# Adding a dataset 
To use the infographicVQA dataset you need to adding the new dataset with following [here](https://mmf.sh/docs/tutorials/dataset). To take less time, you can use files that I created to add the infographicVQA dataset at `https://github.com/Pxtri2156/M4C_inforgraphicsVQA/mmf/datasets/builders/inforgraphicvqa/builder.py` and `https://github.com/Pxtri2156/M4C_inforgraphicsVQA/mmf/datasets/builders/inforgraphicvqa/dataset.py` 

# Prepare annotations
## Create extracted object 
To able run M4C model you prepare extracted features of objects and OCR tokens in images. You follow at [here](https://github.com/facebookresearch/mmf/issues/663#issuecomment-883000371)
## Build annotation file
After having extracted features. We will build the annotation file with following **TextVQA** annotation. 
You can see my annotation files at [here](https://drive.google.com/drive/folders/1th648ZxtUeB-PANpU89usRebxPrudjcB?usp=sharing). I also wrote script to convert **InfographicVQA** annotation format to **TextVQA** annotaions format at [here](https://github.com/Pxtri2156/M4C_inforgraphicsVQA/blob/main/utils/prepare_annotation.py).
# Train and Inference
Now, you can train and inference with M4C.  
To train
```
mmf_run dataset=textvqa \
  model=m4c \
  config=projects/m4c/configs/textvqa/defaults.yaml \
  env.save_dir=./save/m4c
```
To evaluate     
```
mmf_run dataset=textvqa \
  model=m4c \
  config=projects/m4c/configs/textvqa/defaults.yaml \
  env.save_dir=./save/m4c \
  run_type=val \
  checkpoint.resume_zoo=m4c.textvqa.with_stvqa
```