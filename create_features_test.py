#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""特徴量作成のテスト（小規模データ）"""

import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import os
from datetime import datetime
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

def main():
    """メイン処理"""
    # S3の設定
    bucket_name = "fiby-yamasa-prediction"
    input_file_key = "df_confirmed_order_input_yamasa_fill_zero_df_confirmed_order_input_yamasa_fill_zero.csv"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file_key = f"output_data/features/df_features_yamasa_{timestamp}.parquet"
    latest_file_key = "output_data/features/df_features_yamasa_latest.parquet"

    print("="*50)
    print("特徴量作成テスト（小規模データ）")
    print("="*50)

    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))

    try:
        # 小さいサンプルで読み込む（100MB）
        print("サンプルデータ読込中（100MB）...")
        obj = s3.get_object(Bucket=bucket_name, Key=input_file_key, Range='bytes=0-104857600')
        raw_data = obj['Body'].read()

        # UTF-16、タブ区切りで読み込む
        print("データ解析中...")
        df = pd.read_csv(BytesIO(raw_data), encoding='utf-16', sep='\t', on_bad_lines='skip')
        print(f"読込完了: {df.shape}")
        print(f"カラム: {list(df.columns)}")

        # データ型の最適化
        print("\nデータ型最適化中...")
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')

        # 日付処理
        if 'File Date' in df.columns:
            print("日付処理中...")
            df['File Date'] = pd.to_datetime(df['File Date'], errors='coerce')
            df['year'] = df['File Date'].dt.year
            df['month'] = df['File Date'].dt.month
            df['day'] = df['File Date'].dt.day
            df['day_of_week'] = df['File Date'].dt.dayofweek

        # Material Key毎の基本統計量
        if 'Material Key' in df.columns and 'Actual Value' in df.columns:
            print("集計特徴量作成中...")
            agg_features = df.groupby('Material Key')['Actual Value'].agg([
                ('mean_f', 'mean'),
                ('std_f', 'std'),
                ('min_f', 'min'),
                ('max_f', 'max'),
                ('count_f', 'count')
            ]).reset_index()

            df = df.merge(agg_features, on='Material Key', how='left')

            # ラグ特徴量
            df = df.sort_values(['Material Key', 'File Date'])
            df['lag_1_f'] = df.groupby('Material Key')['Actual Value'].shift(1)

        print(f"\n特徴量作成完了: {df.shape}")
        print(f"カラム数: {len(df.columns)}")

        # S3に保存
        print("\nS3への保存中...")
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False, compression='snappy')
        parquet_buffer.seek(0)

        # 保存
        s3.put_object(Bucket=bucket_name, Key=output_file_key, Body=parquet_buffer.getvalue())
        print(f"データ保存成功: s3://{bucket_name}/{output_file_key}")

        s3.put_object(Bucket=bucket_name, Key=latest_file_key, Body=parquet_buffer.getvalue())
        print(f"最新版保存成功: s3://{bucket_name}/{latest_file_key}")

        # メタデータ保存
        metadata = {
            'created_at': timestamp,
            'shape': str(df.shape),
            'columns': list(df.columns)
        }
        metadata_key = "output_data/features/df_features_yamasa_latest_metadata.json"
        s3.put_object(
            Bucket=bucket_name,
            Key=metadata_key,
            Body=pd.Series(metadata).to_json(),
            ContentType='application/json'
        )
        print(f"メタデータ保存成功: s3://{bucket_name}/{metadata_key}")

        print("\n処理完了！")

    except Exception as e:
        print(f"\nエラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()