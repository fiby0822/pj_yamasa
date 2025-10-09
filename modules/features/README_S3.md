# S3統合付き特徴量生成

## デフォルト値の更新
- `train_end_date`: "2024-12-31"（デフォルト）
- `start_year`: 2021（デフォルト）
- `end_year`: 2025（デフォルト）

## S3への保存場所

特徴量は以下の構造でS3に保存されます：

```
s3://fiby-yamasa-prediction/
└── output/
    └── features/
        ├── confirmed_order_demand_yamasa_features_YYYYMMDD_HHMMSS.parquet  # タイムスタンプ付き
        ├── confirmed_order_demand_yamasa_features_latest.parquet            # 最新版（常に更新）
        ├── unofficial_features_YYYYMMDD_HHMMSS.parquet
        ├── unofficial_features_latest.parquet
        ├── use_actual_value_by_category_features_YYYYMMDD_HHMMSS.parquet
        └── use_actual_value_by_category_features_latest.parquet
```

## 最新ファイルの参照

各モデルタイプの最新特徴量は `{model_type}_features_latest.parquet` で常に参照可能：

```python
# 最新の特徴量を読み込む
latest_key = "output/features/confirmed_order_demand_yamasa_features_latest.parquet"
df_latest = generator.load_data_from_s3(latest_key)
```

## 使用方法

### 1. Pythonコードから使用

```python
from modules.features.feature_generator_with_s3 import FeatureGeneratorWithS3

# 初期化
generator = FeatureGeneratorWithS3(bucket_name="fiby-yamasa-prediction")

# S3からデータ読み込み
df_input = generator.load_data_from_s3("input_data/raw_data.parquet")

# 特徴量生成とS3保存（output_keyを省略すると自動生成）
df_features = generator.generate_and_save_features(
    df_input=df_input,
    model_type="confirmed_order_demand_yamasa",
    start_year=2021,      # デフォルト値
    end_year=2025,        # デフォルト値
    train_end_date="2024-12-31",  # デフォルト値
    create_latest=True    # features_latest.parquetも作成（デフォルト）
)

# 最新の特徴量を読み込む場合
df_latest = generator.load_data_from_s3(
    "output/features/confirmed_order_demand_yamasa_features_latest.parquet"
)
```

### 2. コマンドラインから使用

```bash
# 単一モデルの特徴量生成
python3 run_feature_generation.py \
    --input-key data/input/raw_data.parquet \
    --model-type confirmed_order_demand_yamasa \
    --start-year 2021 \
    --end-year 2025 \
    --train-end-date 2024-12-31

# バッチ処理（全モデルタイプを一括処理）
python3 run_feature_generation.py \
    --input-key data/input/raw_data.parquet \
    --batch-process

# CSV形式で保存
python3 run_feature_generation.py \
    --input-key data/input/raw_data.parquet \
    --output-key features/output.csv \
    --save-format csv
```

## S3保存の詳細

### FeatureGeneratorWithS3クラス

`modules/features/feature_generator_with_s3.py`に実装されています：

- **generate_and_save_features()**: 特徴量を生成してS3に自動保存
- **load_data_from_s3()**: S3からデータを読み込み
- **process_batch_models()**: 複数モデルを一括処理

### 保存形式

- Parquet形式（デフォルト、推奨）
- CSV形式（オプション）

### 保存先の命名規則

- 自動生成: `output/features/{model_type}_features_{timestamp}.parquet`
- 最新版: `output/features/{model_type}_features_latest.parquet`（常に同じパスで最新を参照可能）
- カスタム: `--output-key`パラメータで指定可能

## 必要な環境変数

`.env`ファイルに以下を設定：

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-northeast-1
```

## エラーハンドリング

- S3接続エラー時は詳細なエラーメッセージを表示
- データ読み込み失敗時は処理を中断
- 保存成功時はS3のフルパスを表示