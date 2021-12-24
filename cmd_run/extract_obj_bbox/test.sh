FOLDER_PATH_TO_DATASET="/mlcv/Databases/VN_InfographicVQA/test/documents/"  # folder contain image
OUTPUT_FOLDER="env_variable/data/datasets/vi_infographicvqa/200_object/features" #folder obtain npy file 

CUDA_VISIBLE_DEVICES=5 python tools/scripts/features/extract_features_vmb.py \
--model_name="X-101" \
--num_features=200 \
--image_dir=$FOLDER_PATH_TO_DATASET \
--output_folder=$OUTPUT_FOLDER


