LMDB_PATH="env_variable/data/datasets/vi_infographicvqa/defaults/features"
FEATURES_PATH="env_variable/data/datasets/vi_infographicvqa/defaults/features"

python tools/scripts/features/lmdb_conversion.py \
    --mode="convert" \
    --lmdb_path=$LMDB_PATH \
    --features_folder=$FEATURES_PATH
