#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""特徴量作成（ストリーミング処理）"""

import pandas as pd
import numpy as np
import boto3
from smart_open import open as smart_open
import os
from datetime import datetime
from dotenv import load_dotenv
from io import BytesIO

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
    print("特徴量作成（ストリーミング処理）")
    print("="*50)

    # S3クライアント設定
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))

    try:
        # S3 URLを構築
        s3_url = f"s3://{bucket_name}/{input_file_key}"

        print(f"データ読込中: {s3_url}")
        print("※10GBのファイルなので時間がかかります（5-10分）...")

        # smart_openでストリーミング読み込み（チャンク処理）
        chunk_size = 50000  # 一度に処理する行数
        chunks = []

        with smart_open(s3_url, 'r', encoding='utf-16',
                       transport_params={'client': s3}) as f:
            # ヘッダーを読む
            header = f.readline().strip()
            columns = header.split('\t')
            print(f"カラム数: {len(columns)}")
            print(f"カラム（最初の5個）: {columns[:5]}")

            # チャンクごとに読み込み
            lines = []
            line_count = 0
            total_count = 0

            for line in f:
                lines.append(line.strip())
                line_count += 1

                if line_count >= chunk_size:
                    # DataFrameに変換
                    data = [line.split('\t') for line in lines]
                    df_chunk = pd.DataFrame(data, columns=columns)

                    # データ型変換
                    if 'Actual Value' in df_chunk.columns:
                        df_chunk['Actual Value'] = pd.to_numeric(df_chunk['Actual Value'], errors='coerce')

                    chunks.append(df_chunk)
                    total_count += line_count
                    print(f"  処理済み: {total_count:,} 行")

                    # メモリ制限のため、最大100万行まで
                    if total_count >= 1000000:
                        print("  サンプルデータ（100万行）で処理します")
                        break

                    lines = []
                    line_count = 0

            # 残りのデータを処理
            if lines and total_count < 1000000:
                data = [line.split('\t') for line in lines]
                df_chunk = pd.DataFrame(data, columns=columns)
                if 'Actual Value' in df_chunk.columns:
                    df_chunk['Actual Value'] = pd.to_numeric(df_chunk['Actual Value'], errors='coerce')
                chunks.append(df_chunk)
                total_count += line_count

        # チャンクを結合
        print(f"\nデータ結合中...")
        df = pd.concat(chunks, ignore_index=True)
        print(f"データサイズ: {df.shape}")

        # データ型の最適化
        print("データ型最適化中...")
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

            # 数値型に確実に変換
            df['Actual Value'] = pd.to_numeric(df['Actual Value'], errors='coerce').fillna(0)

            agg_features = df.groupby('Material Key')['Actual Value'].agg([
                ('mean_f', 'mean'),
                ('std_f', 'std'),
                ('min_f', 'min'),
                ('max_f', 'max'),
                ('count_f', 'count')
            ]).reset_index()

            df = df.merge(agg_features, on='Material Key', how='left')

            # ラグ特徴量
            print("ラグ特徴量作成中...")
            df = df.sort_values(['Material Key', 'File Date'])
            df['lag_1_f'] = df.groupby('Material Key')['Actual Value'].shift(1)
            df['lag_2_f'] = df.groupby('Material Key')['Actual Value'].shift(2)
            df['lag_3_f'] = df.groupby('Material Key')['Actual Value'].shift(3)

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
        return df

    except Exception as e:
        print(f"\nエラー発生: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()