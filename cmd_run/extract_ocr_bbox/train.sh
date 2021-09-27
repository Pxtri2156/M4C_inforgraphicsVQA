FOLDER_PATH_TO_DATASET="/mlcv/Databases/DocVQA_2020-21/task_3/train/infographicVQA_train_v1.0_images"
IMDB_FILE="my_features/annotations/train/lmdb_val_en_0.0.npy"
DETECTION_MODEl="models_ocr/detectron_model.pth"
DETECTION_CONFIG="models_ocr/detectron_model.yaml"
OUTPUT="my_features/bbox_features/ocr"

CUDA_DEVICES_VISBLE=2 python projects/m4c/scripts/extract_ocr_frcn_feature.py \
    --image_dir=$FOLDER_PATH_TO_DATASET \
    --imdb_file=$IMDB_FILE \
    --detection_model=$DETECTION_MODEl \
    --detection_cfg=$DETECTION_CONFIG \
    --save_dir=$OUTPUT 

# CUDA_DEVICES_VISBLE=2 python projects/m4c/scripts/extract_ocr_frcn_feature.py \
#     --image_dir="/mlcv/Databases/DocVQA_2020-21/task_3/train/infographicVQA_train_v1.0_images" \
#     --imdb_file="env_variable/data/datasets/inforgraphicvqa/defaults/annotations/vqa_train_en.npy" \
#     --detection_model="models_ocr/detectron_model.pth" \
#     --detection_cfg="models_ocr/detectron_model.yaml"\
#     --save_dir="my_features/bbox_features/ocr"