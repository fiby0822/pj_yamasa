#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特徴量作成用スクリプト（UTF-16対応版）
"""

import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import json

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

    print("=" * 50)
    print("特徴量作成処理開始")
    print("=" * 50)

    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))

    try:
        # S3からデータ読込（UTF-16エンコーディング、タブ区切り）
        print(f"\n1. データ読込中...")
        print(f"  ファイル: {input_file_key}")
        print(f"  エンコーディング: UTF-16")
        print(f"  区切り文字: タブ")

        obj = s3.get_object(Bucket=bucket_name, Key=input_file_key)

        # UTF-16でデコードしてDataFrameに読み込む
        # メモリ効率のため、必要最小限の型を指定
        df = pd.read_csv(
            obj['Body'],
            encoding='utf-16',
            sep='\t',
            low_memory=False
        )

        print(f"  読込成功: {df.shape}")
        print(f"  カラム数: {len(df.columns)}")
        print(f"  カラム名（最初の10個）: {list(df.columns)[:10]}")

        # カラム名を正規化（スペースをアンダースコアに）
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

        # 必要なカラムの存在確認と作成
        if 'material_key' not in df.columns:
            if 'material_code' in df.columns:
                df['material_key'] = df['material_code']
            elif 'product_code' in df.columns:
                df['material_key'] = df['product_code']
            elif 'product_key' in df.columns:
                df['material_key'] = df['product_key']
            else:
                # 最初のカラムを使用
                df['material_key'] = df.iloc[:, 0]

        if 'file_date' in df.columns:
            df['file_date'] = pd.to_datetime(df['file_date'], errors='coerce')

        if 'actual_value' not in df.columns:
            # 数値カラムを探す
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                # 'value'を含むカラムを優先
                value_cols = [col for col in numeric_cols if 'value' in col.lower() or 'amount' in col.lower() or 'quantity' in col.lower()]
                if value_cols:
                    df['actual_value'] = df[value_cols[0]]
                else:
                    df['actual_value'] = df[numeric_cols[0]]
            else:
                df['actual_value'] = 0

        print(f"\n2. 特徴量作成中...")

        # データ型の最適化
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')

        # 基本的な時系列特徴量を追加
        if 'file_date' in df.columns:
            df['year_f'] = df['file_date'].dt.year.astype('int16')
            df['month_f'] = df['file_date'].dt.month.astype('int8')
            df['day_f'] = df['file_date'].dt.day.astype('int8')
            df['day_of_week_mon1_f'] = df['file_date'].dt.dayofweek.astype('int8')
            df['quarter_f'] = df['file_date'].dt.quarter.astype('int8')

        # material_key毎の基本統計量
        if 'material_key' in df.columns and 'actual_value' in df.columns:
            print("  集計特徴量作成中...")

            # グループ毎の統計量
            agg_stats = df.groupby('material_key')['actual_value'].agg([
                ('mean_f', 'mean'),
                ('std_f', lambda x: x.std()),
                ('min_f', 'min'),
                ('max_f', 'max'),
                ('median_f', 'median'),
                ('count_f', 'count')
            ]).reset_index()

            df = df.merge(agg_stats, on='material_key', how='left')

            # ラグ特徴量（メモリ効率のため最小限）
            if 'file_date' in df.columns:
                df = df.sort_values(['material_key', 'file_date'])

                # ラグ特徴量
                for lag in [1, 2, 3]:
                    df[f'lag_{lag}_f'] = df.groupby('material_key')['actual_value'].shift(lag)

                # 移動平均（過去の値のみ使用）
                for window in [3, 6]:
                    df[f'rolling_mean_{window}_f'] = df.groupby('material_key')['actual_value'].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                    )

        print(f"  特徴量作成完了: {df.shape}")
        print(f"  特徴量数: {len([col for col in df.columns if col.endswith('_f')])}")

        # S3に保存
        print(f"\n3. データ保存中...")

        # Parquetファイルとして保存
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False, compression='snappy')
        parquet_data = parquet_buffer.getvalue()

        # タイムスタンプ付きで保存
        s3.put_object(Bucket=bucket_name, Key=output_file_key, Body=parquet_data)
        print(f"  保存成功: s3://{bucket_name}/{output_file_key}")

        # 最新版として保存
        s3.put_object(Bucket=bucket_name, Key=latest_file_key, Body=parquet_data)
        print(f"  最新版保存: s3://{bucket_name}/{latest_file_key}")

        # メタデータ保存
        metadata = {
            "created_at": timestamp,
            "input_file": input_file_key,
            "output_file": output_file_key,
            "latest_file": latest_file_key,
            "shape": list(df.shape),
            "n_features": len([col for col in df.columns if col.endswith('_f')]),
            "n_records": len(df),
            "n_materials": df['material_key'].nunique() if 'material_key' in df.columns else 0,
            "columns": list(df.columns)[:50]  # 最初の50カラムのみ
        }

        metadata_json = json.dumps(metadata, indent=2, default=str, ensure_ascii=False)
        s3.put_object(Bucket=bucket_name, Key=f"output_data/features/metadata_{timestamp}.json", Body=metadata_json)
        s3.put_object(Bucket=bucket_name, Key="output_data/features/metadata_latest.json", Body=metadata_json)
        print(f"  メタデータ保存: s3://{bucket_name}/output_data/features/metadata_latest.json")

        print("\n" + "=" * 50)
        print("特徴量作成処理完了")
        print("=" * 50)

        print(f"\n次のコマンドで学習・予測を実行してください:")
        print(f"  source venv/bin/activate && python train_predict.py")

        return True

    except Exception as e:
        print(f"\nエラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)