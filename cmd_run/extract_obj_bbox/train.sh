FOLDER_PATH_TO_DATASET="/mlcv/Databases/DocVQA_2020-21/task_3/train/infographicVQA_train_v1.0_images"
OUTPUT_FOLDER="my_features/bbox_features/object/train"

python tools/scripts/features/extract_features_vmb.py \
--model_name="X-101" \
--image_dir=$FOLDER_PATH_TO_DATASET \
--output_folder=$OUTPUT_FOLDER