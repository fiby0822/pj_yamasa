#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""データ形式確認用スクリプト"""

import pandas as pd
import boto3
from io import StringIO
import os
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

def test_csv_encoding():
    """CSVファイルのエンコーディングとカラムを確認"""

    bucket_name = "fiby-yamasa-prediction"
    file_key = "df_confirmed_order_input_yamasa_fill_zero_df_confirmed_order_input_yamasa_fill_zero.csv"

    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))

    print(f"バケット: {bucket_name}")
    print(f"ファイル: {file_key}")
    print("="*50)

    try:
        # 最初の1MBだけ読み込んでテスト
        print("1. ファイルの先頭部分を取得中...")
        obj = s3.get_object(Bucket=bucket_name, Key=file_key, Range='bytes=0-1048576')
        raw_data = obj['Body'].read()

        # エンコーディングの検出を試す
        print("\n2. エンコーディング検出中...")
        encodings = ['utf-8', 'shift-jis', 'cp932', 'euc-jp', 'iso-2022-jp', 'latin-1']

        for enc in encodings:
            try:
                text_data = raw_data.decode(enc)
                lines = text_data.split('\n')[:5]
                print(f"\n✓ {enc} でデコード成功:")
                print("  最初の5行:")
                for i, line in enumerate(lines):
                    if len(line) > 100:
                        print(f"  {i+1}: {line[:100]}...")
                    else:
                        print(f"  {i+1}: {line}")

                # pandasで読み込みテスト
                df = pd.read_csv(StringIO(text_data), nrows=10)
                print(f"\n  カラム数: {len(df.columns)}")
                print(f"  カラム名: {list(df.columns)[:10]}")
                print(f"  データ型:\n{df.dtypes.head()}")
                print(f"  データサイズ: {df.shape}")

                return enc

            except Exception as e:
                print(f"✗ {enc} でエラー: {str(e)[:50]}")
                continue

        print("\n全てのエンコーディングで失敗しました")

    except Exception as e:
        print(f"エラー発生: {e}")

if __name__ == "__main__":
    encoding = test_csv_encoding()