# 特徴量生成モジュール

時系列データから特徴量を生成するモジュールです。`train_end_date`パラメータをサポートし、データリークを防ぎます。

## 主な機能

- **train_end_date対応**: 指定日以降のデータを欠損値として扱い、未来のデータを使わない特徴量生成
- **複数モデルタイプ対応**:
  - `confirmed_order_demand_yamasa`: ヤマサ確定注文予測用
  - `unofficial`: 内示予測用
  - `use_actual_value_by_category`: カテゴリベース予測用
- **多様な時系列特徴量**:
  - ラグ特徴量
  - 移動平均・移動標準偏差
  - 累積平均
  - 変動率
  - 重み付き累積平均（カテゴリモデル用）

## 使用方法

```python
from modules.features.timeseries_features import add_timeseries_features
from modules.config.feature_window_config import WINDOW_SIZE_CONFIG

# 特徴量生成
df_features = add_timeseries_features(
    df_input,
    window_size_config=WINDOW_SIZE_CONFIG,
    start_year=2024,
    end_year=2024,
    model_type="confirmed_order_demand_yamasa",
    train_end_date="2024-09-30"  # この日付より後のデータは使用しない
)
```

## パラメータ

- `_df`: 入力データフレーム（material_key, file_date, actual_valueを含む）
- `window_size_config`: 特徴量のウィンドウサイズ設定（省略時はデフォルト設定使用）
- `start_year`, `end_year`: 出力データの年範囲
- `model_type`: モデルタイプ
- `train_end_date`: 学習データの終了日（この日付より後のデータは欠損値として扱う）

## 設定ファイル

ウィンドウサイズの設定は `modules/config/feature_window_config.py` で管理されています。
各モデルタイプに応じて異なる特徴量が生成されます。

## テスト

```bash
python3 test_feature_generation.py
```

テストスクリプトでは、train_end_dateが正しく機能していることを確認できます。