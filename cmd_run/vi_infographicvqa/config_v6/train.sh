MMF_SAVE_DIR="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/save/vi_infographicVQA/config_v6"
MMF_DATA_DIR="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/data"
MMF_CACHE_DIR="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/cache"
MMF_LOG_DIR="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/log/vi_infographicVQA/config_v6"
MMF_USER_DIR="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/user"
MMF_TENSORBOARD_LOGDIR="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/tensorboard/vi_infographicVQA/config_v6"
MMF_REPORT_DIR="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/report/vi_infographicVQA/config_v6"

CUDA_VISIBLE_DEVICES=6 mmf_run config=projects/m4c/configs/vi_infographicvqa/100_object.yaml \
    datasets=vi_infographicvqa \
    model=m4c \
    env.save_dir=$MMF_SAVE_DIR \
    env.data_dir=$MMF_DATA_DIR \
    env.cache_dir=$MMF_CACHE_DIR \
    env.log_dir=$MMF_LOG_DIR \
    env.user_dir=$MMF_USER_DIR \
    env.tensorboard_logdir=$MMF_TENSORBOARD_LOGDIR \
    env.report_dir=$MMF_REPORT_DIR \
    run_type=train_val \
    training.batch_size=16 \
    training.num_workers=0 \
    training.checkpoint_interval=500 \
    training.evaluation_interval=500 \
    checkpoint.resume=True \
    training.tensorboard=True \
    training.max_updates=6000