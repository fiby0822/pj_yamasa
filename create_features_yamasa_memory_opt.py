#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""特徴量作成（メモリ最適化版）- usage_type別処理"""

import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import os
from datetime import datetime
from dotenv import load_dotenv
import jpholiday
import gc

# 環境変数を読み込み
load_dotenv()

# データ型を最小限に
dtype_dict = {
    'material_key': 'category',
    'product_key': 'category',
    'store_code': 'category',
    'usage_type': 'category',
    'product_name': 'category',
    'category_lvl1': 'category',
    'category_lvl2': 'category',
    'category_lvl3': 'category',
    'container': 'category',
    'actual_value': 'float32',
    'day_of_week_mon1': 'int8',
    'week_number': 'int8',
    'month': 'int8'
}

def reduce_mem_usage(df):
    """メモリ使用量を削減"""
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        # 日付型はスキップ
        if col_type == 'datetime64[ns]' or col_type == 'datetime64[ns, UTC]':
            continue

        if col_type != object and str(col_type) != 'category':
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'  メモリ削減: {start_mem:.1f} MB → {end_mem:.1f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}%減)')

    return df

def create_basic_features(df):
    """基本的な特徴量のみを作成（メモリ効率重視）"""
    print("  基本特徴量を作成中...")

    # 日付特徴量
    df['year'] = df['file_date'].dt.year.astype('int16')
    df['month'] = df['file_date'].dt.month.astype('int8')
    df['day'] = df['file_date'].dt.day.astype('int8')
    df['day_of_week'] = df['file_date'].dt.dayofweek.astype('int8')
    df['week_of_year'] = df['file_date'].dt.isocalendar().week.astype('int8')
    df['quarter'] = df['file_date'].dt.quarter.astype('int8')

    # 週末フラグ
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')

    # 月初・月末フラグ
    df['is_month_start'] = df['file_date'].dt.is_month_start.astype('int8')
    df['is_month_end'] = df['file_date'].dt.is_month_end.astype('int8')

    # サイン・コサイン変換
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype('float32')
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype('float32')

    return df

def create_lag_features_by_usage(df, usage_type, lags=[1, 2, 3, 7]):
    """usage_type毎にラグ特徴量を作成"""
    print(f"    {usage_type}のラグ特徴量を作成中...")

    # usage_typeでフィルタ
    df_usage = df[df['usage_type'] == usage_type].copy()

    # material_key毎にソート
    df_usage = df_usage.sort_values(['material_key', 'file_date'])

    # ラグ特徴量（簡易版）
    for lag in lags:
        df_usage[f'lag_{lag}'] = df_usage.groupby('material_key')['actual_value'].shift(lag)

    # 7日移動平均（簡易版）
    df_usage['ma_7'] = df_usage.groupby('material_key')['actual_value'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    ).astype('float32')

    # 14日移動平均
    df_usage['ma_14'] = df_usage.groupby('material_key')['actual_value'].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    ).astype('float32')

    return df_usage

def main():
    """メイン処理"""
    # S3の設定
    bucket_name = "fiby-yamasa-prediction"
    input_file_key = "data/df_confirmed_order_input_yamasa_fill_zero.parquet"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ローカル保存先
    output_dir = "output_data/features"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/df_features_yamasa_{timestamp}.parquet"
    latest_file = f"{output_dir}/df_features_yamasa_latest.parquet"

    print("="*50)
    print("ヤマサ確定注文用特徴量作成（メモリ最適化版）")
    print("="*50)

    # S3クライアント設定
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

    try:
        print(f"データ読込中: s3://{bucket_name}/{input_file_key}")

        # 必要なカラムのみ読み込み
        essential_cols = [
            'material_key', 'file_date', 'actual_value',
            'usage_type', 'product_key', 'store_code',
            'category_lvl1', 'category_lvl2', 'category_lvl3',
            'container', 'product_name'
        ]

        # S3からParquetファイルを読み込み
        obj = s3.get_object(Bucket=bucket_name, Key=input_file_key)
        df = pd.read_parquet(BytesIO(obj['Body'].read()), columns=essential_cols)

        print(f"読込データサイズ: {df.shape[0]:,} 行 × {df.shape[1]} 列")
        print(f"メモリ使用量: {df.memory_usage().sum() / 1024**2:.1f} MB")

        # データ型最適化
        for col, dtype in dtype_dict.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        # メモリ削減
        df = reduce_mem_usage(df)
        gc.collect()

        # 日付型変換
        df['file_date'] = pd.to_datetime(df['file_date'])

        print("\n特徴量作成を開始...")

        # 基本特徴量（全データ共通）
        df = create_basic_features(df)
        gc.collect()

        # usage_type別に処理
        usage_types = df['usage_type'].unique()
        df_list = []

        for usage_type in usage_types:
            print(f"\n  {usage_type}の特徴量作成中...")
            df_usage = create_lag_features_by_usage(df, usage_type)
            df_list.append(df_usage)

            # メモリ解放
            del df_usage
            gc.collect()

        # 結合
        print("\n結果を結合中...")
        df_final = pd.concat(df_list, ignore_index=True)

        # メモリ解放
        del df, df_list
        gc.collect()

        # 再度メモリ最適化
        df_final = reduce_mem_usage(df_final)

        print(f"\n最終データサイズ: {df_final.shape[0]:,} 行 × {df_final.shape[1]} 列")
        print(f"最終メモリ使用量: {df_final.memory_usage().sum() / 1024**2:.1f} MB")

        # usage_type毎の統計
        if 'usage_type' in df_final.columns:
            print("\nusage_type別統計:")
            for usage_type in df_final['usage_type'].unique():
                subset = df_final[df_final['usage_type'] == usage_type]
                print(f"  {usage_type}: {len(subset):,} 行")

        # ローカルに保存
        print(f"\n保存中...")
        df_final.to_parquet(output_file, index=False, compression='snappy')
        df_final.to_parquet(latest_file, index=False, compression='snappy')
        print(f"  ローカル保存完了: {output_file}")

        # S3に保存
        print("S3に保存中...")

        # タイムスタンプ付きファイル
        with open(output_file, 'rb') as f:
            s3.put_object(
                Bucket=bucket_name,
                Key=f"features/df_features_yamasa_{timestamp}.parquet",
                Body=f.read()
            )
            print(f"  S3保存完了: s3://{bucket_name}/features/df_features_yamasa_{timestamp}.parquet")

        # 最新版ファイル
        with open(latest_file, 'rb') as f:
            s3.put_object(
                Bucket=bucket_name,
                Key=f"features/df_features_yamasa_latest.parquet",
                Body=f.read()
            )
            print(f"  S3保存完了: s3://{bucket_name}/features/df_features_yamasa_latest.parquet")

        print("\n特徴量作成完了!")
        return df_final

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()