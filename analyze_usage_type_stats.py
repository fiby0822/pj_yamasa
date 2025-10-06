#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
usage_type毎のmaterial_key数を計算
"""

import pandas as pd
import boto3
import os
from io import BytesIO
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

def get_s3_client():
    """S3クライアントを取得"""
    return boto3.client('s3',
                       aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                       aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                       region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

def main():
    """メイン処理"""
    print("="*70)
    print("usage_type毎のmaterial_key数を計算")
    print("="*70)

    # S3設定
    bucket_name = "fiby-yamasa-prediction"
    s3 = get_s3_client()

    # S3からファイルを読み込み
    input_key = "data/df_confirmed_order_input_yamasa_fill_zero.parquet"

    print(f"\nファイルを読み込み中: s3://{bucket_name}/{input_key}")

    try:
        obj = s3.get_object(Bucket=bucket_name, Key=input_key)
        df = pd.read_parquet(BytesIO(obj['Body'].read()))
        print(f"✓ 読込成功: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return

    # 基本統計
    print("\n【基本情報】")
    print(f"  総レコード数: {len(df):,}")
    print(f"  総material_key数: {df['material_key'].nunique():,}")

    # usage_type毎の統計
    print("\n【usage_type別のmaterial_key数】")
    print("-" * 50)

    if 'usage_type' in df.columns:
        # usage_type毎のユニークなmaterial_key数
        usage_stats = df.groupby('usage_type').agg({
            'material_key': lambda x: x.nunique(),
            'actual_value': [
                lambda x: len(x),  # レコード数
                lambda x: (x == 0).sum(),  # ゼロの数
                lambda x: (x > 0).sum(),  # 非ゼロの数
            ]
        })

        usage_stats.columns = ['unique_material_keys', 'total_records', 'zero_records', 'nonzero_records']

        total_keys = df['material_key'].nunique()
        total_records = len(df)

        for usage_type, row in usage_stats.iterrows():
            keys = row['unique_material_keys']
            records = row['total_records']
            zeros = row['zero_records']
            nonzeros = row['nonzero_records']

            print(f"\n{usage_type}:")
            print(f"  - material_key数: {keys:,} ({keys/total_keys*100:.1f}% of total)")
            print(f"  - レコード数: {records:,} ({records/total_records*100:.1f}% of total)")
            print(f"    - ゼロ値: {zeros:,} ({zeros/records*100:.1f}%)")
            print(f"    - 非ゼロ値: {nonzeros:,} ({nonzeros/records*100:.1f}%)")

        print("\n" + "-" * 50)
        print("\n【合計】")
        print(f"  - 総material_key数: {total_keys:,}")
        print(f"  - 総レコード数: {total_records:,}")
        print(f"  - 総ゼロ値: {(df['actual_value'] == 0).sum():,} ({(df['actual_value'] == 0).sum()/total_records*100:.1f}%)")
        print(f"  - 総非ゼロ値: {(df['actual_value'] > 0).sum():,} ({(df['actual_value'] > 0).sum()/total_records*100:.1f}%)")

        # material_keyの重複チェック
        print("\n【material_keyの重複チェック】")
        key_usage = df.groupby('material_key')['usage_type'].nunique()
        duplicated_keys = key_usage[key_usage > 1]

        if len(duplicated_keys) > 0:
            print(f"  ⚠ {len(duplicated_keys):,} 個のmaterial_keyが複数のusage_typeに存在")
            print("\n  重複例（最初の5件）:")
            for key in duplicated_keys.head().index:
                types = df[df['material_key'] == key]['usage_type'].unique()
                print(f"    - {key}: {', '.join(types)}")
        else:
            print("  ✓ 重複なし（各material_keyは1つのusage_typeのみに存在）")

        # 期間の確認
        print("\n【期間】")
        df['file_date'] = pd.to_datetime(df['file_date'])
        for usage_type in df['usage_type'].unique():
            sub = df[df['usage_type'] == usage_type]
            print(f"  {usage_type}: {sub['file_date'].min().date()} ～ {sub['file_date'].max().date()}")

    else:
        print("  ✗ usage_type列が存在しません")

    print("\n" + "="*70)
    print("完了!")
    print("="*70)

if __name__ == "__main__":
    main()