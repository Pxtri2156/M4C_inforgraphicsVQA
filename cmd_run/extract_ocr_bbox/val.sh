FOLDER_PATH_TO_DATASET="/mlcv/Databases/VN_InfographicVQA/val/documents/"
IMDB_FILE="env_variable/data/datasets/vi_infographicvqa/defaults/annotations/infoVQA_val_vi.npy"
DETECTION_MODEl="models_ocr/detectron_model.pth"
DETECTION_CONFIG="models_ocr/detectron_model.yaml"
OUTPUT="env_variable/data/datasets/vi_infographicvqa/ocr_vi/features"

CUDA_VISIBLE_DEVICES=6 python projects/m4c/scripts/extract_ocr_frcn_feature.py \
    --image_dir=$FOLDER_PATH_TO_DATASET \
    --imdb_file=$IMDB_FILE \
    --detection_model=$DETECTION_MODEl \
    --detection_cfg=$DETECTION_CONFIG \
    --save_dir=$OUTPUT 

