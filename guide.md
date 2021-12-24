# Create enviroment
## Create docker container
```
docker run --ipc=host -it --gpus all --runtime=nvidia --name <container_name> \
   --cpus=30 --shm-size=512m  \
   -m 30000M --memory-swap -1 \
   -v /mlcv:/mlcv -v /home:/home \
   nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 /bin/bash
```
Example: 
```
docker run --ipc=host -it --gpus all --runtime=nvidia --name mlcv_pxtri_docvqa_mmf_v1 \
   --cpus=30 --shm-size=512m  \
   -m 30000M --memory-swap -1 \
   -v /mlcv:/mlcv -v /home:/home \
   nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 /bin/bash
```
## Exec container docker 
```
docker star mlcv_pxtri_docvqa_mmf_v1
docker exec -it mlcv_pxtri_docvqa_mmf_v1 /bin/bash;
```
## Create environtment in conda 
## Activate conda environment 
Example
```
source /mlcv/WorkingSpace/SceneText/tripx/anaconda3/bin/activate; 
conda activate mmf_docVQA
```
# Test code 
We test code with TextVQA dataset. However, you have to set environment variable. 
```
mmf_user_dir = 
mmf_cache_dir = 
mmf_data_dir = #where obtain data that include annotation 
mmf_log_dir = #where you save logfile
mmf_save_dir = #where you save model
mmf_tensorboard_dir = 
mmf_wandb_log_dir = 

```
Then you can try to run code
```
CUDA_VISIBLE_DEVICES=0 mmf_run config=projects/m4c/configs/textvqa/defaults.yaml \
    datasets=textvqa \
    model=m4c \
    env.save_dir=$MMF_SAVE_DIR \
    env.data_dir=$MMF_DATA_DIR \
    env.cache_dir=$MMF_CACHE_DIR \
    env.log_dir=$MMF_LOG_DIR \
    env.user_dir=$MMF_USER_DIR \
    env.tensorboard_logdir=$MMF_TENSORBOARD_LOGDIR \
    env.report_dir=$MMF_REPORT_DIR \
    run_type=train_val \
    training.batch_size=8 \
    training.num_workers=1
``` 
# Adding a dataset 
## Build dataset builder
## Build dataset class 
## Build metric for dataset

# Prepare annotations
## Create extracted object 
Activate env `maskrcnn_benchmark_v3`
```
FOLDER_PATH_TO_DATASET="/mlcv/Databases/VN_InfographicVQA/train/documents/"  # folder contain image
OUTPUT_FOLDER="env_variable/data/datasets/vi_infographicvqa/defaults/features" #folder obtain npy file 

CUDA_VISIBLE_DEVICES=2 python tools/scripts/features/extract_features_vmb.py \
--model_name="X-101" \
--num_features=30 \
--image_dir=$FOLDER_PATH_TO_DATASET \
--output_folder=$OUTPUT_FOLDER
```