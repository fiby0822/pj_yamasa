#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""特徴量作成（confirmed_order_demand_yamasa版）"""

import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import os
from datetime import datetime
from dotenv import load_dotenv
import jpholiday

# 環境変数を読み込み
load_dotenv()

# window_size_configの定義（ノートブックと同じ設定）
window_size_config = {
    "material_key": {
        "lag": [1,2,3],
        "window": [7, 14, 28],
        "stats": ["mean", "max", "min", "std", "trend"]
    }
}

def create_lag_features(df, config, target_col='actual_value'):
    """ラグ特徴量の作成"""
    for key, params in config.items():
        if key not in df.columns:
            continue

        print(f"  {key}のラグ特徴量を作成中...")

        # ラグ特徴量
        for lag in params.get('lag', []):
            col_name = f'{target_col}_lag_{lag}'
            df[col_name] = df.groupby(key)[target_col].shift(lag)

        # ウィンドウ統計量
        for window in params.get('window', []):
            for stat in params.get('stats', []):
                col_name = f'{target_col}_{stat}_{window}d'

                if stat == 'mean':
                    df[col_name] = df.groupby(key)[target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                elif stat == 'max':
                    df[col_name] = df.groupby(key)[target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
                elif stat == 'min':
                    df[col_name] = df.groupby(key)[target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).min()
                    )
                elif stat == 'std':
                    df[col_name] = df.groupby(key)[target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                elif stat == 'trend':
                    # 簡単なトレンド（現在値 - window日前の値）
                    df[col_name] = df.groupby(key)[target_col].transform(
                        lambda x: x - x.shift(window)
                    )

    return df

def create_date_features(df, date_col='file_date'):
    """日付特徴量の作成"""
    print("  日付特徴量を作成中...")

    # 日付型に変換
    df[date_col] = pd.to_datetime(df[date_col])

    # 基本的な日付特徴量
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['quarter'] = df[date_col].dt.quarter

    # 月初・月末フラグ
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)

    # 週末フラグ
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # 祝日フラグ（日本の祝日）
    print("  祝日情報を追加中...")
    df['is_holiday'] = df[date_col].apply(lambda x: jpholiday.is_holiday(x)).astype(int)

    # 祝日の前後フラグ
    df['is_day_before_holiday'] = df['is_holiday'].shift(-1).fillna(0).astype(int)
    df['is_day_after_holiday'] = df['is_holiday'].shift(1).fillna(0).astype(int)

    # サイン・コサイン変換（周期性を表現）
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df

def create_product_features(df):
    """商品関連の特徴量作成"""
    print("  商品関連特徴量を作成中...")

    # カテゴリのラベルエンコーディング
    categorical_cols = ['category_lvl1', 'category_lvl2', 'category_lvl3', 'container']

    for col in categorical_cols:
        if col in df.columns:
            # NaNを'unknown'で置換
            df[col] = df[col].fillna('unknown')
            # ラベルエンコーディング
            df[f'{col}_encoded'] = pd.Categorical(df[col]).codes

    # volumeを数値型に変換
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['volume'] = df['volume'].fillna(df['volume'].median())

    return df

def create_store_features(df):
    """店舗関連の特徴量作成"""
    print("  店舗関連特徴量を作成中...")

    if 'store_code' in df.columns:
        # 店舗ごとの統計量（過去の平均出荷数など）
        store_stats = df.groupby('store_code')['actual_value'].agg(['mean', 'std', 'median']).reset_index()
        store_stats.columns = ['store_code', 'store_mean_shipment', 'store_std_shipment', 'store_median_shipment']

        # マージ
        df = df.merge(store_stats, on='store_code', how='left')

    return df

def create_interaction_features(df):
    """相互作用特徴量の作成"""
    print("  相互作用特徴量を作成中...")

    # 曜日×月の相互作用
    df['day_of_week_month'] = df['day_of_week'] * 10 + df['month']

    # 週末×月の相互作用
    df['is_weekend_month'] = df['is_weekend'] * df['month']

    # 商品×店舗の組み合わせ統計
    if 'product_key' in df.columns and 'store_code' in df.columns:
        # 商品×店舗の過去平均
        product_store_mean = df.groupby(['product_key', 'store_code'])['actual_value'].mean().reset_index()
        product_store_mean.columns = ['product_key', 'store_code', 'product_store_mean']
        df = df.merge(product_store_mean, on=['product_key', 'store_code'], how='left')

    return df

def add_usage_type_features(df):
    """usage_type関連の特徴量追加"""
    print("  usage_type関連特徴量を作成中...")

    if 'usage_type' in df.columns:
        # usage_typeのラベルエンコーディング
        df['usage_type_encoded'] = pd.Categorical(df['usage_type']).codes

        # usage_type別の統計量
        usage_stats = df.groupby('usage_type')['actual_value'].agg(['mean', 'std', 'median']).reset_index()
        usage_stats.columns = ['usage_type', 'usage_mean_shipment', 'usage_std_shipment', 'usage_median_shipment']
        df = df.merge(usage_stats, on='usage_type', how='left')

    return df

def main():
    """メイン処理"""
    # S3の設定
    bucket_name = "fiby-yamasa-prediction"
    input_file_key = "data/df_confirmed_order_input_yamasa_fill_zero.parquet"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


    print("="*50)
    print("ヤマサ確定注文用特徴量作成")
    print("="*50)

    # S3クライアント設定
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

    try:
        print(f"データ読込中: s3://{bucket_name}/{input_file_key}")

        # S3からParquetファイルを直接読み込み
        obj = s3.get_object(Bucket=bucket_name, Key=input_file_key)
        df = pd.read_parquet(BytesIO(obj['Body'].read()))

        print(f"読込データサイズ: {df.shape[0]:,} 行 × {df.shape[1]} 列")

        # カラム名の確認
        print(f"\nカラム一覧:")
        print(df.columns.tolist()[:15])  # 最初の15カラムを表示

        # データ型の確認と調整
        if 'actual_value' in df.columns:
            df['actual_value'] = pd.to_numeric(df['actual_value'], errors='coerce')

        # 日付型への変換
        if 'file_date' in df.columns:
            df['file_date'] = pd.to_datetime(df['file_date'])
            # 日付でソート
            df = df.sort_values(['material_key', 'file_date']).reset_index(drop=True)

        print("\n特徴量作成を開始...")

        # 1. 日付特徴量
        df = create_date_features(df)
        print(f"  日付特徴量作成完了: {df.shape[1]} カラム")

        # 2. ラグ特徴量と統計量
        df = create_lag_features(df, window_size_config)
        print(f"  ラグ特徴量作成完了: {df.shape[1]} カラム")

        # 3. 商品関連特徴量
        df = create_product_features(df)
        print(f"  商品特徴量作成完了: {df.shape[1]} カラム")

        # 4. 店舗関連特徴量
        df = create_store_features(df)
        print(f"  店舗特徴量作成完了: {df.shape[1]} カラム")

        # 5. 相互作用特徴量
        df = create_interaction_features(df)
        print(f"  相互作用特徴量作成完了: {df.shape[1]} カラム")

        # 6. usage_type関連特徴量
        df = add_usage_type_features(df)
        print(f"  usage_type特徴量作成完了: {df.shape[1]} カラム")

        print(f"\n最終データサイズ: {df.shape[0]:,} 行 × {df.shape[1]} 列")

        # 特徴量の統計情報
        print("\n=== 特徴量統計 ===")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"数値特徴量数: {len(numeric_cols)}")

        # usage_type毎の統計
        if 'usage_type' in df.columns:
            print("\nusage_type別統計:")
            for usage_type in df['usage_type'].unique():
                subset = df[df['usage_type'] == usage_type]
                print(f"  {usage_type}: {len(subset):,} 行")

        # S3に保存
        print("\nS3に保存中...")

        # Parquetバッファを作成
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)

        # タイムスタンプ付きファイル
        s3.put_object(
            Bucket=bucket_name,
            Key=f"features/df_features_yamasa_{timestamp}.parquet",
            Body=parquet_buffer.getvalue()
        )
        print(f"  S3保存完了: s3://{bucket_name}/features/df_features_yamasa_{timestamp}.parquet")

        # 最新版ファイル
        parquet_buffer.seek(0)
        s3.put_object(
            Bucket=bucket_name,
            Key=f"features/df_features_yamasa_latest.parquet",
            Body=parquet_buffer.getvalue()
        )
        print(f"  S3保存完了: s3://{bucket_name}/features/df_features_yamasa_latest.parquet")

        print("\n特徴量作成完了!")
        return df

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()