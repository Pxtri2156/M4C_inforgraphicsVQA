LMDB_PATH="env_variable/data/datasets/inforgraphicvqa/defaults/features"
FEATURES_PATH="my_features/bbox_features/object"

python tools/scripts/features/lmdb_conversion.py \
    --mode="convert" \
    --lmdb_path=$LMDB_PATH \
    --features_folder=$FEATURES_PATH
