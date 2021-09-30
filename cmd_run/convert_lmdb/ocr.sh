LMDB_PATH="env_variable/data/datasets/inforgraphicvqa/ocr_en/features"
FEATURES_PATH="env_variable/data/datasets/inforgraphicvqa/ocr_en/features"

python tools/scripts/features/lmdb_conversion.py \
    --mode="convert" \
    --lmdb_path=$LMDB_PATH \
    --features_folder=$FEATURES_PATH
