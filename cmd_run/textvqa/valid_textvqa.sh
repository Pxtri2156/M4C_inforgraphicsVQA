#environment variable
MMF_SAVE_DIR="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/save"
# MMF_DATA_DIR="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/data"
MMF_DATA_DIR="/mlcv/Databases/DocVQA_2020-21/task_3/extracted_features/m4c/data"
MMF_CACHE_DIR="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/cache"
MMF_LOG_DIR="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/log"
MMF_USER_DIR="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/user"
MMF_TENSORBOARD_LOGDIR="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/tensorboard"
MMF_REPORT_DIR="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/report"



CUDA_VISIBLE_DEVICES=1 mmf_run dataset=textvqa \
   model=m4c \
   config=projects/m4c/configs/textvqa/defaults.yaml \
   env.save_dir=$MMF_SAVE_DIR \
   env.data_dir=$MMF_DATA_DIR \
   env.cache_dir=$MMF_CACHE_DIR \
   env.log_dir=$MMF_LOG_DIR \
   env.user_dir=$MMF_USER_DIR \
   env.tensorboard_logdir=$MMF_TENSORBOARD_LOGDIR \
   env.report_dir=$MMF_REPORT_DIR \
   run_type=val  \
   checkpoint.resume_zoo=m4c.textvqa.with_stvqa