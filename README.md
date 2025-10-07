# ヤマサ確定注文需要予測システム

## 概要
このシステムは、S3に格納されたデータを使用して確定注文の需要予測を行います。

## ファイル構成
- `create_features.py` - 特徴量作成スクリプト
- `train_predict.py` - 学習・予測実行スクリプト
- `run_pipeline.sh` - 全体パイプライン実行スクリプト
- `.env` - AWS認証情報（セキュリティに注意）

## セットアップ

### 1. 仮想環境の有効化
```bash
source venv/bin/activate
```

### 2. 必要なライブラリのインストール（既にインストール済み）
```bash
pip install boto3 python-dotenv xgboost scikit-learn optuna
```

## 実行方法

### 方法1: 全体パイプラインの実行（推奨）
```bash
# デフォルト: 特徴量作成 → 学習・予測（最新の保存済み特徴量を使用）
./run_pipeline.sh

# 特徴量作成をスキップして、保存済みの最新特徴量で学習・予測のみ実行
./run_pipeline.sh --skip-features

# 特定の特徴量ファイルを指定して実行
./run_pipeline.sh --skip-features --features-file "output_data/features/df_features_yamasa_20241201_120000.parquet"
```

### 方法2: 個別実行

#### 1. 特徴量作成
```bash
python create_features.py
```
- 入力: S3上の `df_confirmed_order_input_yamasa_fill_zero.csv`
- 出力:
  - S3の `output_data/features/df_features_yamasa_TIMESTAMP.parquet` （タイムスタンプ版）
  - S3の `output_data/features/df_features_yamasa_latest.parquet` （最新版）
  - メタデータJSONファイル

#### 2. 学習・予測
##### 必須要件(学習・予測時)
 - train_start_dateまでの期間を用いてモデルの学習を行う
 - step_countで指定した月の数だけ予測を行う｡例えばstep_count=6,train_start_date=2025/12/31の場合､2025/1~2025/6までの予測結果を返す
 - 予測対象のデータに対して､実績値がゼロより大きい値に対して予測する
 - モデル実行時は下記項目を表示する｡モデルや特徴量の変更があった場合は､変更前と変更後の値を表示する｡誤差率の定義は「(予測値-実績値の絶対値)/実績値」をmaterial_key毎に計算する
  - RMSE
  - MAE
  - 誤差率平均値
  - 誤差率中央値
  - 誤差率が20%以内のmaterial_key数・割合
  - 誤差率が30%以内のmaterial_key数・割合
  - 誤差率が50%以内のmaterial_key数・割合

##### 必須要件(S3出力時)
 - 出力先ディレクトリは`output_data/predictions/`に保存し､最新の実行結果を`_latest`というsuffixをつけて保存する
 - 特徴量重要度(feature_importance)は全件出力する

##### 保存済みの最新特徴量を使用（デフォルト）
```bash
python train_predict.py
# または明示的に指定
python train_predict.py --is-use-saved-features
```

##### 特定の特徴量ファイルを指定
```bash
python train_predict.py \
    --no-saved-features \
    --features-file "output_data/features/df_features_yamasa_20241201_120000.parquet" \
    --train-end-date "2024-12-31" \
    --step-count 6 \
    --use-optuna \
    --n-trials 50
```
- 入力: 特徴量Parquetファイル（保存済み最新版または指定ファイル）
- 出力: S3の `output_data/predictions/` ディレクトリに予測結果

## S3構造
```
s3://fiby-yamasa-prediction/
├── df_confirmed_order_input_yamasa_fill_zero.csv  # 入力データ
└── output_data/
    ├── features/
    │   └── df_features_yamasa_*.parquet          # 特徴量データ
    └── predictions/
        ├── predictions_*.parquet                  # 予測結果
        ├── bykey_errors_*.csv                    # material_key別エラー率
        ├── feature_importance_*.csv              # 特徴量重要度
        ├── params_*.json                         # 最適パラメータ
        └── metrics_*.json                        # 評価メトリクス
```

## パラメータ説明

### create_features.py
- `window_size_config`: 時系列特徴量の設定（ラグ、移動平均、累積平均など）
- `start_year`, `end_year`: データの対象期間
- `model_type`: "confirmed_order_demand_yamasa"を指定

### train_predict.py
- `--features-file`: 特徴量ファイルのS3キー
- `--train-end-date`: 学習データの終了日
- `--step-count`: 予測ステップ数（月単位）
- `--use-optuna`: Optunaでのパラメータ最適化を使用
- `--n-trials`: Optunaの試行回数

## 注意事項
- AWS認証情報は`.env`ファイルに保存されています
- S3へのアクセスにはap-northeast-1リージョンを使用
- ローカルにも`output/`ディレクトリに結果のコピーが保存されます

## トラブルシューティング
1. S3アクセスエラー: AWS認証情報を確認
2. メモリ不足: データサイズを確認、必要に応じてインスタンスサイズを増やす
3. ライブラリエラー: 仮想環境が有効化されているか確認