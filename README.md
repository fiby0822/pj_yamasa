# ヤマサ確定注文需要予測システム

## 概要
このシステムは、S3に格納されたExcelデータを使用して確定注文の需要予測を行います。
時系列データの特徴量生成、XGBoostによる機械学習モデルの構築、Walk-forward validationによる評価を実装しています。

## セットアップ

### 1. 仮想環境の有効化
```bash
source venv/bin/activate
```

### 2. 必要なライブラリのインストール
```bash
pip install -r requirements.txt
```

### 必要なパッケージ
- boto3 (S3アクセス)
- pandas, numpy (データ処理)
- xgboost (機械学習モデル)
- optuna (ハイパーパラメータ最適化)
- scikit-learn (評価指標)
- matplotlib, seaborn (可視化)
- pyarrow (Parquetファイル処理)

## ディレクトリ構成
```
yamasa/
├── modules/                # コアモジュール
│   ├── core/              # データ準備のコアモジュール
│   │   ├── prepare/       # データ準備処理
│   │   └── archive/       # 過去のファイル
│   ├── config/            # 設定ファイル
│   │   └── feature_window_config.py  # 特徴量ウィンドウサイズ設定
│   ├── features/          # 特徴量生成モジュール
│   │   ├── timeseries_features.py    # 時系列特徴量生成
│   │   └── feature_generator_with_s3.py  # S3連携特徴量生成
│   ├── models/            # 機械学習モジュール
│   │   ├── train_predict.py  # モデル学習・予測
│   │   └── predictor.py      # 予測実行
│   ├── evaluation/        # 評価モジュール
│   │   └── metrics.py        # 評価指標計算
│   ├── data_io/          # データI/O
│   │   └── s3_handler.py     # S3ハンドラー
│   └── metrics/          # モデル比較・評価
├── scripts/              # 実行スクリプト
│   ├── prepare/         # データ準備スクリプト
│   │   ├── run_prepare_yamasa.py      # Excelファイル加工
│   │   └── run_prepare_fill_zero.py   # ゼロ値補完
│   ├── features/        # 特徴量生成スクリプト
│   │   ├── generate_features_batch_s3.py  # バッチ処理（S3中間保存）
│   │   └── generate_features_yamasa.py    # 通常処理
│   ├── train/          # モデル学習スクリプト
│   │   └── train_model.py            # モデル学習実行
│   └── predict/        # 予測スクリプト
│       └── run_prediction.py         # 予測実行
├── notebooks/          # Jupyter Notebook格納
├── tests/             # テストスクリプト
├── logs/              # ログファイル
├── requirements.txt   # 依存パッケージ
├── run_prepare.sh     # データ準備実行シェル
└── run_train_predict.sh  # 学習・予測実行シェル
```

## 処理概要

### 1. データの取得・加工 (prepare_data_**.py)
1-1. S3に置かれているエクセルファイルを加工・結合し、S3に保存する
   - 入力: `s3://fiby-yamasa-prediction/input_data/*.xlsx`
   - 出力ファイル名: `df_confirmed_order_input_yamasa.parquet`

1-2. 1-1のデータはactual_value(実績値)が発生しているレコードしかないため、実績値がゼロのレコードを保存する必要がある
   - material_key毎にfile_dateのminとmaxを取得し、その間の期間に対して、actual_value=0のレコードを挿入する
   - 他の項目はfile_dateから遡って最新のレコードを補完する
   - 出力ファイル名: `df_confirmed_order_input_yamasa_fill_zero.parquet`

### 2. 特徴量の生成 (create_features_**.py)
1のデータをもとに特徴量を生成する
- 特徴量はラグ特徴量やmaterial_key毎の週平均など、過去データを用いるものが多いが、train_end_dateで指定した期間の特徴量を用いる
- 即ち、train_end_dateより後の期間のactual_valueは欠損値として特徴量を全期間に対して計算する
- 生成される特徴量:
  - ラグ特徴量（1, 2, 3期前）
  - 移動平均（2, 3, 6期間）
  - 移動標準偏差（2, 3, 6期間）
  - 累積平均（2, 3, 6, 12期間）
  - 変化率
  - 曜日、週番号、月、年の特徴量
- 出力: `confirmed_order_demand_yamasa_features_latest.parquet`

### 3. 予測モデルの構築・予測の実行 (train_predict_**.py)
- 2で生成した特徴量を用いる
- step_countで指定した月の数だけ予測を行う
  - 例: step_count=6, train_end_date=2024/12/31の場合、2025/1~2025/6までの6ヶ月分の予測結果を返す
- 予測対象のデータに対して、実績値がゼロより大きい値に対して予測する
- モデルは指示がなければxgboostを使い、ランダムサンプリングは行わない
- Walk-forward validation（月単位）で検証
- 外れ値処理: Winsorization、Hampelフィルタを適用

### 4. 評価指標を算出し、表示・保存 (analyze_**.py)
モデル実行と同じ処理内で、下記項目を計算し表示・保存する。モデルや特徴量の変更があった場合は、変更前と変更後の値を表示する。

誤差率の定義: 「(予測値-実績値の絶対値)/実績値」をmaterial_key毎に計算する

- **基本評価指標**:
  - RMSE
  - MAE
  - 誤差率平均値
  - 誤差率中央値

- **誤差率分析**:
  - 誤差率が20%以内のmaterial_key数・割合
  - 誤差率が30%以内のmaterial_key数・割合
  - 誤差率が50%以内のmaterial_key数・割合

- **追加指標**:
  - MAPE（平均絶対パーセント誤差）
  - 相関係数
  - R²スコア

- **Material Key別評価**: 各material_keyの予測精度を個別に評価
- **可視化**: 実績vs予測散布図、誤差分布、時系列グラフ

## 実行方法

### 方法1: 全体パイプラインの実行

#### データ準備
```bash
./run_prepare.sh
```

#### モデル学習・予測（基本）
```bash
./run_train_predict.sh train
```

#### モデル学習・予測（Optuna最適化付き）
```bash
./run_train_predict.sh train true
```

#### フルパイプライン（準備＋学習＋予測）
```bash
./run_train_predict.sh full
```

### 方法2: 個別実行

#### データ準備
```bash
# Excelファイルの加工
python3 scripts/prepare/run_prepare_yamasa.py

# ゼロ値補完
python3 scripts/prepare/run_prepare_fill_zero.py
```

#### 特徴量生成
```bash
# 通常処理（メモリ32GB以上推奨）
python3 scripts/features/generate_features_yamasa.py

# バッチ処理（メモリ制限がある場合）
python3 scripts/features/generate_features_batch_s3.py
```

#### モデル学習
```bash
# 基本学習（1ヶ月予測）
python3 scripts/train/train_model.py \
    --train-end-date "2024-12-31" \
    --step-count 1

# Optuna最適化付き学習（6ヶ月予測）
python3 scripts/train/train_model.py \
    --train-end-date "2024-12-31" \
    --step-count 6 \
    --use-optuna \
    --n-trials 50
```

#### 予測実行
```bash
# 将来予測
python3 scripts/predict/run_prediction.py \
    --mode future \
    --start-date "2025-01-01" \
    --end-date "2025-01-31" \
    --save-results
```

## S3構造
```
s3://fiby-yamasa-prediction/
├── input_data/                    # 入力Excelファイル
│   └── *.xlsx
└── output/                        # 処理結果
    ├── df_confirmed_order_input_yamasa.parquet           # 加工済みデータ
    ├── df_confirmed_order_input_yamasa_fill_zero.parquet # ゼロ値補完済み
    ├── features/                  # 特徴量データ
    │   ├── confirmed_order_demand_yamasa_features_latest.parquet
    │   ├── confirmed_order_demand_yamasa_features_[timestamp].parquet
    │   └── temp_batches/          # バッチ処理の中間ファイル
    ├── models/                    # 学習済みモデル
    │   ├── confirmed_order_demand_yamasa_model_latest.pkl
    │   ├── confirmed_order_demand_yamasa_params_latest.pkl
    │   └── confirmed_order_demand_yamasa_model_[timestamp].pkl
    ├── predictions/               # 予測結果
    │   ├── confirmed_order_demand_yamasa_predictions_latest.parquet
    │   └── confirmed_order_demand_yamasa_predictions_[timestamp].parquet
    └── evaluation/                # 評価結果
        ├── confirmed_order_demand_yamasa_metrics_latest.json
        ├── confirmed_order_demand_yamasa_predictions_latest.csv
        └── *.png                  # 可視化画像
```

## パラメータ説明

### feature_window_config.py
```python
WINDOW_SIZE_CONFIG = {
    "material_key": {
        "lag": [1, 2, 3],           # ラグ期間
        "rolling_mean": [2, 3, 6],  # 移動平均ウィンドウ
        "rolling_std": [2, 3, 6],   # 移動標準偏差ウィンドウ
        "cumulative_mean": [2, 3, 6, 12],  # 累積平均期間
    },
    # store_code, usage_type等も同様に設定
}
```

### train_model.py
- `--train-end-date`: 学習データの終了日（YYYY-MM-DD）
- `--step-count`: 予測月数（1~12）
- `--use-optuna`: Optunaでのハイパーパラメータ最適化
- `--n-trials`: Optunaの試行回数（デフォルト: 50）
- `--no-outlier-handling`: 外れ値処理を無効化
- `--features-path`: 特徴量ファイルのS3パス
- `--save-dir`: モデル保存先のS3パス

### run_prediction.py
- `--mode`: 予測モード（future, walk-forward, material-key, single-date）
- `--start-date`, `--end-date`: 予測期間
- `--material-keys`: 対象Material Keyリスト
- `--aggregate`: 集約方法（sum, mean, median）
- `--save-results`: 結果をS3に保存

## 性能・制限事項

### メモリ要件
- 全データ処理: 32GB以上推奨
- バッチ処理モード: 16GBでも動作可能
- データサイズ: 約3,200万レコード、38,512 material keys

### 処理時間（r6a.xlarge: 32GB RAM, 4 vCPU）
- データ準備: 約5-10分
- 特徴量生成: 約30-60分（バッチ処理）
- モデル学習: 約10-30分（データサイズによる）

### 主要な機能
- ✅ 月単位のWalk-forward validation
- ✅ テストデータのみでの評価（過学習防止）
- ✅ train_end_dateによるデータリーク防止
- ✅ Material Key別の詳細評価
- ✅ S3との完全統合（ローカル保存なし）
- ✅ メモリ効率的なバッチ処理
- ✅ Optuna によるハイパーパラメータ最適化

## 注意事項
- AWS認証情報は環境変数またはIAMロールで管理
- S3へのアクセスにはap-northeast-1リージョンを使用
- ローカルへの保存は行わず、全てS3に保存
- データ量が大きいため、処理実行前に必ずリソースを確認

## Claude Codeにおける注意点
- データ量が非常に大きいため、「処理を実行してください」と明示的に指示されない限りコマンドは実行しない
- S3に新しくデータを出力する際は出力先パスを必ず表示する
- 新しくPythonファイルを作る場合はファイル名を表示し、処理概要をREADMEに追記する
- メモリ不足の場合はバッチ処理モードを推奨する
- **重要**: サンプリングによる実行及び実装は一切行わない。全データでの処理が必要な場合は、メモリ増設やバッチ処理などの他の解決策を提案する

## トラブルシューティング

### メモリ不足エラー
```bash
# バッチ処理モードで実行
python3 scripts/features/generate_features_batch_s3.py
```

### S3アクセスエラー
```bash
# AWS認証情報の確認
aws sts get-caller-identity
```

### 特徴量が見つからない
```bash
# 特徴量生成を実行
python3 scripts/features/generate_features_yamasa.py
```

## 更新履歴
- 2024/10/09: Walk-forward validation実装、月単位予測対応
- 2024/10/09: バッチ処理によるメモリ効率化
- 2024/10/09: train_end_dateによるデータリーク防止機能追加
- 2024/10/09: Optuna統合、評価指標の詳細化