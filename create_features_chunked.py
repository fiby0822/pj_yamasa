#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特徴量作成用スクリプト（大規模データ対応版）
S3からデータをチャンクで読み込み、時系列特徴量を追加して保存する
"""

import pandas as pd
import numpy as np
import boto3
from io import StringIO, BytesIO
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

# メモリ効率のための設定
CHUNK_SIZE = 100000  # 一度に処理する行数

def read_csv_sample_from_s3(bucket_name, file_key, nrows=1000):
    """S3からCSVの先頭部分を読み込む（データ構造確認用）"""
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))
    try:
        # 最初の部分だけ読み込む（UTF-16なので多めに取得）
        obj = s3.get_object(Bucket=bucket_name, Key=file_key, Range=f'bytes=0-20971520')  # 20MB
        raw_data = obj['Body'].read()
        # UTF-16、タブ区切りで読み込む
        df_sample = pd.read_csv(BytesIO(raw_data), encoding='utf-16', sep='\t', nrows=nrows, on_bad_lines='skip')
        print(f"サンプルデータ読込成功: {df_sample.shape}")
        return df_sample
    except Exception as e:
        print(f"S3からの読み込みエラー: {e}")
        raise

def process_features_simple(df):
    """シンプルな特徴量作成（メモリ効率重視）"""
    df = df.copy()

    # データ型の最適化
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            max_val = df[col].max()
            min_val = df[col].min()
            if min_val >= 0 and max_val <= 255:
                df[col] = df[col].astype('uint8')
            elif min_val >= -32768 and max_val <= 32767:
                df[col] = df[col].astype('int16')
            elif min_val >= -2147483648 and max_val <= 2147483647:
                df[col] = df[col].astype('int32')

    # 日付処理（カラム名を正しく変換）
    if 'File Date' in df.columns:
        df['File Date'] = pd.to_datetime(df['File Date'], errors='coerce')
        df['year'] = df['File Date'].dt.year.astype('int16')
        df['month'] = df['File Date'].dt.month.astype('int8')
        df['day'] = df['File Date'].dt.day.astype('int8')
        df['day_of_week'] = df['File Date'].dt.dayofweek.astype('int8')

        # 基本的な特徴量を追加
        df['year_f'] = df['year']
        df['month_f'] = df['month']
        df['day_f'] = df['day']
        df['day_of_week_mon1_f'] = df['day_of_week']

    return df

def main():
    """メイン処理"""
    # S3の設定
    bucket_name = "fiby-yamasa-prediction"
    input_file_key = "df_confirmed_order_input_yamasa_fill_zero_df_confirmed_order_input_yamasa_fill_zero.csv"  # 正しいファイル名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file_key = f"output_data/features/df_features_yamasa_{timestamp}.parquet"
    latest_file_key = "output_data/features/df_features_yamasa_latest.parquet"

    print("=" * 50)
    print("特徴量作成処理開始（軽量版）")
    print("=" * 50)

    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))

    try:
        # まずサンプルデータで構造を確認
        print(f"\n1. データ構造確認中...")
        df_sample = read_csv_sample_from_s3(bucket_name, input_file_key, nrows=1000)
        print(f"カラム: {list(df_sample.columns)[:10]}...")
        print(f"データ型: {df_sample.dtypes.head()}")

        # material_key毎の集計を行う
        print(f"\n2. material_key毎の集計処理...")

        # S3から全データを読み込む（メモリに注意）
        print("全データ読込中（これには時間がかかる場合があります）...")
        obj = s3.get_object(Bucket=bucket_name, Key=input_file_key)
        # UTF-16、タブ区切りで読み込む
        df = pd.read_csv(obj['Body'], encoding='utf-16', sep='\t', low_memory=False)
        print(f"読込完了: {df.shape}")

        # データ型の最適化
        print("データ型最適化中...")
        df = process_features_simple(df)

        # Material Key毎の基本統計量を作成（正しいカラム名を使用）
        if 'Material Key' in df.columns and 'Actual Value' in df.columns:
            print("集計特徴量作成中...")

            # Material Key毎の基本統計量
            agg_features = df.groupby('Material Key')['Actual Value'].agg([
                ('mean_f', 'mean'),
                ('std_f', 'std'),
                ('min_f', 'min'),
                ('max_f', 'max'),
                ('count_f', 'count')
            ]).reset_index()

            # メインデータフレームにマージ
            df = df.merge(agg_features, on='Material Key', how='left')

            # シンプルなラグ特徴量（メモリ効率のため最小限）
            if 'File Date' in df.columns:
                df = df.sort_values(['Material Key', 'File Date'])
                df['lag_1_f'] = df.groupby('Material Key')['Actual Value'].shift(1)
                df['lag_2_f'] = df.groupby('Material Key')['Actual Value'].shift(2)
                df['lag_3_f'] = df.groupby('Material Key')['Actual Value'].shift(3)

        print(f"特徴量作成後のデータサイズ: {df.shape}")

        # S3に保存
        print(f"\n3. データ保存中...")
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False, compression='snappy')
        parquet_buffer.seek(0)

        # タイムスタンプ付きで保存
        s3.put_object(Bucket=bucket_name, Key=output_file_key, Body=parquet_buffer.getvalue())
        print(f"データ保存成功: s3://{bucket_name}/{output_file_key}")

        # 最新版として保存
        s3.put_object(Bucket=bucket_name, Key=latest_file_key, Body=parquet_buffer.getvalue())
        print(f"最新版保存成功: s3://{bucket_name}/{latest_file_key}")

        # メタデータ保存
        metadata = {
            "created_at": timestamp,
            "input_file": input_file_key,
            "output_file": output_file_key,
            "latest_file": latest_file_key,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "feature_columns": [col for col in df.columns if col.endswith('_f')],
            "n_features": len([col for col in df.columns if col.endswith('_f')]),
            "n_records": len(df),
            "n_materials": df['material_key'].nunique() if 'material_key' in df.columns else 0
        }

        import json
        metadata_json = json.dumps(metadata, indent=2, default=str)
        s3.put_object(Bucket=bucket_name, Key=f"output_data/features/metadata_{timestamp}.json", Body=metadata_json)
        s3.put_object(Bucket=bucket_name, Key="output_data/features/metadata_latest.json", Body=metadata_json)

        print("\n" + "=" * 50)
        print("特徴量作成処理完了")
        print("=" * 50)
        print(f"\n特徴量数: {len([col for col in df.columns if col.endswith('_f')])}")
        print(f"次のステップで使用するファイル:")
        print(f"  - 最新版: {latest_file_key}")
        print(f"  - タイムスタンプ版: {output_file_key}")

        return True

    except Exception as e:
        print(f"\nエラー発生: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)