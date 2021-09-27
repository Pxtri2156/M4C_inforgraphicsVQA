FOLDER_PATH_TO_DATASET="./"
OUTPUT="lmdb_ocr/textVQA"
IMDB_FILE="env_variable/data/datasets/textvqa/defaults/annotations/imdb_test_ocr_en.npy"
DETECTION_MODEl="models_ocr/detectron_model.pth"
DETECTION_CONFIG="models_ocr/detectron_model.yaml"

CUDA_DEVICES_VISIBLE=1 python projects/m4c/scripts/extract_ocr_frcn_feature.py \
    --image_dir=$FOLDER_PATH_TO_DATASET \
    --imdb_file=$IMDB_FILE \
    --detection_model=$DETECTION_MODEl \
    --detection_cfg=$DETECTION_CONFIG \
    --save_dir=$OUTPUT 