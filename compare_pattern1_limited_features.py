#!/usr/bin/env python3
"""
パターン1: 特徴量限定版（12特徴量のみ、Optuna無し）
ゼロ補完済みデータから読み込み
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import boto3
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# サンプリング設定
SAMPLE_SIZE = 500000  # 50万行でテスト
RANDOM_STATE = 42

# 指定された12個の特徴量
SPECIFIED_FEATURES = [
    'year_f',
    'month_f',
    'lag_1_f',
    'lag_2_f',
    'lag_3_f',
    'rolling_mean_2_f',
    'rolling_mean_3_f',
    'rolling_mean_6_f',
    'cumulative_mean_2_f',
    'cumulative_mean_3_f',
    'cumulative_mean_6_f',
    'cumulative_mean_12_f'
]

def create_limited_features(df):
    """指定された12個の特徴量のみを作成"""
    print("="*50)
    print("特徴量作成（12個限定版）")
    print("="*50)

    # カラム名の正規化
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # 日付処理
    if 'file_date' in df.columns:
        df['file_date'] = pd.to_datetime(df['file_date'], errors='coerce')

    # 数値変換
    if 'actual_value' in df.columns:
        df['actual_value'] = pd.to_numeric(df['actual_value'], errors='coerce').fillna(0)

    # ソート（時系列特徴量のため重要）
    if 'material_key' in df.columns and 'file_date' in df.columns:
        df = df.sort_values(['material_key', 'file_date'])

    features = []

    # 1. 日付特徴量（year_f, month_f）
    if 'file_date' in df.columns:
        print("日付特徴量作成中...")
        df['year_f'] = df['file_date'].dt.year
        df['month_f'] = df['file_date'].dt.month
        features.extend(['year_f', 'month_f'])

    # 2. material_keyごとの時系列特徴量
    if 'material_key' in df.columns and 'actual_value' in df.columns:
        print("時系列特徴量作成中...")
        df_grouped = df.groupby('material_key')['actual_value']

        # Lag特徴量 (lag_1_f, lag_2_f, lag_3_f)
        for lag in [1, 2, 3]:
            feature_name = f'lag_{lag}_f'
            df[feature_name] = df_grouped.shift(lag).fillna(0)
            features.append(feature_name)

        # Rolling Mean (rolling_mean_2_f, rolling_mean_3_f, rolling_mean_6_f)
        for window in [2, 3, 6]:
            feature_name = f'rolling_mean_{window}_f'
            df[feature_name] = df_grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            ).fillna(0)
            features.append(feature_name)

        # Cumulative Mean (cumulative_mean_2_f, cumulative_mean_3_f, cumulative_mean_6_f, cumulative_mean_12_f)
        for window in [2, 3, 6, 12]:
            feature_name = f'cumulative_mean_{window}_f'
            df[feature_name] = df_grouped.transform(
                lambda x: x.expanding(min_periods=min(window, 1)).mean()
            ).fillna(0)
            features.append(feature_name)

    # 作成された特徴量を確認
    created_features = [f for f in SPECIFIED_FEATURES if f in df.columns]
    missing_features = [f for f in SPECIFIED_FEATURES if f not in df.columns]

    print(f"\n作成された特徴量: {len(created_features)}個")
    if missing_features:
        print(f"作成できなかった特徴量: {missing_features}")

    return df, created_features

def train_model_without_optuna(X_train, y_train, X_test, y_test):
    """Optuna無しでモデルを学習（固定パラメータ）"""
    print("\n" + "="*50)
    print("モデル学習（Optuna無し、固定パラメータ）")
    print("="*50)

    # 固定パラメータ
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

    print("モデルパラメータ:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")

    # モデル学習
    print("\nモデル学習中...")
    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)

    # 予測
    print("予測中...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 評価
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    print("\n訓練データの評価:")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  MAE: {train_mae:.2f}")
    print(f"  R2: {train_r2:.4f}")

    print("\nテストデータの評価:")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE: {test_mae:.2f}")
    print(f"  R2: {test_r2:.4f}")

    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n特徴量重要度（上位5）:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    metrics = {
        'pattern': 'パターン1（12特徴量、Optuna無し）',
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'n_features': len(X_train.columns),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }

    return model, metrics, feature_importance

def main():
    print("="*70)
    print("パターン1: 特徴量限定版（12特徴量、Optuna無し）")
    print("="*70)

    # S3設定
    bucket_name = "fiby-yamasa-prediction"
    input_key = "data/df_confirmed_order_input_yamasa_fill_zero.parquet"

    print(f"\nデータ読込中: s3://{bucket_name}/{input_key}")

    # データ読み込み
    df = pd.read_parquet(f"s3://{bucket_name}/{input_key}")
    print(f"読込完了: {len(df):,} 行")

    # サンプリング
    if len(df) > SAMPLE_SIZE:
        print(f"\nサンプリング実施: {SAMPLE_SIZE:,} 行")
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    # 特徴量作成
    df, feature_cols = create_limited_features(df)

    # データ準備
    X = df[feature_cols]
    y = df['actual_value']

    # 無限大と欠損値の処理
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    # データ分割
    print(f"\nデータ分割中...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=False
    )

    print(f"  訓練データ: {len(X_train):,} 行")
    print(f"  テストデータ: {len(X_test):,} 行")
    print(f"  特徴量数: {len(feature_cols)} 個")

    # モデル学習と評価
    model, metrics, importance = train_model_without_optuna(X_train, y_train, X_test, y_test)

    # 結果保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # メトリクスをCSV保存
    metrics_df = pd.DataFrame([metrics])
    metrics_file = f"pattern1_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\nメトリクス保存: {metrics_file}")

    # 特徴量重要度をCSV保存
    importance_file = f"pattern1_importance_{timestamp}.csv"
    importance.to_csv(importance_file, index=False)
    print(f"特徴量重要度保存: {importance_file}")

    print("\n" + "="*70)
    print("パターン1 完了")
    print("="*70)

    return metrics

if __name__ == "__main__":
    metrics = main()