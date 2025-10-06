#!/usr/bin/env python3
"""
パターン1用の特徴量生成スクリプト
ゼロ補完済みデータから12個の特徴量を生成
"""

import pandas as pd
import numpy as np
import boto3
import os
from datetime import datetime
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

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

def create_features(df):
    """12個の特徴量を生成"""
    print("特徴量生成開始...")

    # カラム名の正規化
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # 日付と数値の処理
    df['file_date'] = pd.to_datetime(df['file_date'], errors='coerce')
    df['actual_value'] = pd.to_numeric(df['actual_value'], errors='coerce').fillna(0)

    # ソート（重要）
    print("データをソート中...")
    df = df.sort_values(['material_key', 'file_date'])

    features = []

    # 1. 日付特徴量
    print("日付特徴量作成中...")
    df['year_f'] = df['file_date'].dt.year
    df['month_f'] = df['file_date'].dt.month
    features.extend(['year_f', 'month_f'])

    # 2. material_keyごとの時系列特徴量
    print("時系列特徴量作成中...")
    df_grouped = df.groupby('material_key')['actual_value']

    # Lag特徴量（1, 2, 3日前）
    print("  - Lag特徴量...")
    for lag in [1, 2, 3]:
        feature_name = f'lag_{lag}_f'
        df[feature_name] = df_grouped.shift(lag).fillna(0)
        features.append(feature_name)

    # Rolling Mean（2, 3, 6日間の移動平均）
    print("  - Rolling Mean特徴量...")
    for window in [2, 3, 6]:
        feature_name = f'rolling_mean_{window}_f'
        df[feature_name] = df_grouped.transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        ).fillna(0)
        features.append(feature_name)

    # Cumulative Mean（2, 3, 6, 12期間の累積平均）
    print("  - Cumulative Mean特徴量...")
    for window in [2, 3, 6, 12]:
        feature_name = f'cumulative_mean_{window}_f'
        df[feature_name] = df_grouped.transform(
            lambda x: x.expanding(min_periods=min(window, 1)).mean()
        ).fillna(0)
        features.append(feature_name)

    # 作成された特徴量の確認
    created_features = [f for f in SPECIFIED_FEATURES if f in df.columns]
    print(f"\n作成された特徴量: {len(created_features)}個")
    print(f"特徴量リスト: {created_features}")

    return df, created_features

def main():
    """メイン処理"""
    print("="*70)
    print("パターン1 特徴量生成")
    print("="*70)

    # S3設定
    bucket_name = "fiby-yamasa-prediction"
    input_key = "data/df_confirmed_order_input_yamasa_fill_zero.parquet"

    # S3クライアント
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

    # データ読み込み
    print(f"\nデータ読込中: s3://{bucket_name}/{input_key}")
    df = pd.read_parquet(f"s3://{bucket_name}/{input_key}")
    print(f"読込完了: {len(df):,} 行 × {len(df.columns)} 列")

    # メモリ使用量の確認
    memory_usage = df.memory_usage(deep=True).sum() / 1024**3
    print(f"メモリ使用量: {memory_usage:.2f} GB")

    # データ基本情報
    print("\n=== データ基本情報 ===")
    if 'actual_value' in df.columns:
        print(f"actual_valueの統計:")
        print(f"  平均: {df['actual_value'].mean():.2f}")
        print(f"  標準偏差: {df['actual_value'].std():.2f}")
        print(f"  ゼロの割合: {(df['actual_value'] == 0).mean()*100:.1f}%")

    if 'file_date' in df.columns:
        df['file_date'] = pd.to_datetime(df['file_date'], errors='coerce')
        print(f"\nfile_dateの範囲:")
        print(f"  開始: {df['file_date'].min()}")
        print(f"  終了: {df['file_date'].max()}")

    # 特徴量生成
    print("\n" + "="*70)
    print("特徴量生成")
    print("="*70)
    df_features, feature_cols = create_features(df)

    # 必要なカラムのみ保持
    keep_cols = ['material_key', 'store_code', 'product_key', 'file_date', 'actual_value'] + feature_cols
    keep_cols = [col for col in keep_cols if col in df_features.columns]
    df_final = df_features[keep_cols]

    # 保存
    print("\n" + "="*70)
    print("結果保存")
    print("="*70)

    # ローカル保存
    os.makedirs("output_data/features", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    local_file = f"output_data/features/df_features_pattern1_{timestamp}.parquet"
    df_final.to_parquet(local_file, index=False, compression='snappy')
    print(f"✓ ローカル保存: {local_file}")

    # 最新版も保存
    latest_file = "output_data/features/df_features_pattern1_latest.parquet"
    df_final.to_parquet(latest_file, index=False, compression='snappy')
    print(f"✓ 最新版: {latest_file}")

    # S3保存
    print("\nS3にアップロード中...")
    buffer = BytesIO()
    df_final.to_parquet(buffer, index=False, compression='snappy')
    buffer.seek(0)

    # タイムスタンプ付きファイル
    s3_key = f"output/features/df_features_pattern1_{timestamp}.parquet"
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=buffer.getvalue())
    print(f"✓ S3保存: s3://{bucket_name}/{s3_key}")

    # 最新版
    buffer.seek(0)
    s3_latest_key = "output/features/df_features_pattern1_latest.parquet"
    s3.put_object(Bucket=bucket_name, Key=s3_latest_key, Body=buffer.getvalue())
    print(f"✓ S3最新版: s3://{bucket_name}/{s3_latest_key}")

    print("\n" + "="*70)
    print("特徴量生成完了")
    print("="*70)
    print(f"\n最終データ:")
    print(f"  行数: {len(df_final):,}")
    print(f"  列数: {len(df_final.columns)}")
    print(f"  特徴量数: {len(feature_cols)}")

    return df_final

if __name__ == "__main__":
    df = main()