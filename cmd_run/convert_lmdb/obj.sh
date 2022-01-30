LMDB_PATH="env_variable/data/datasets/vi_infographicvqa/200_object/features"
FEATURES_PATH="env_variable/data/datasets/vi_infographicvqa/200_object/features"

python tools/scripts/features/lmdb_conversion.py \
    --mode="convert" \
    --lmdb_path=$LMDB_PATH \
    --features_folder=$FEATURES_PATH
