#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UTF-16形式のデータ確認用スクリプト"""

import pandas as pd
import boto3
import os
from dotenv import load_dotenv
from io import BytesIO

# 環境変数を読み込み
load_dotenv()

def test_utf16_csv():
    """UTF-16形式のCSVファイルを確認"""

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
        # 最初の10MBだけ読み込んでテスト
        print("ファイルの先頭部分を取得中...")
        obj = s3.get_object(Bucket=bucket_name, Key=file_key, Range='bytes=0-10485760')
        raw_data = obj['Body'].read()

        # UTF-16でデコード（部分的なデータなのでエラーを無視）
        print("\nUTF-16でデコード中...")
        text_data = raw_data.decode('utf-16', errors='ignore')

        # 最初の数行を表示
        lines = text_data.split('\n')[:5]
        print("\n最初の5行:")
        for i, line in enumerate(lines):
            if len(line) > 100:
                print(f"  {i+1}: {line[:100]}...")
            else:
                print(f"  {i+1}: {line}")

        # pandasで読み込み
        print("\npandasで読み込み中...")
        # UTF-16のバイトデータから直接読み込む
        df = pd.read_csv(BytesIO(raw_data), encoding='utf-16', sep='\t', nrows=100)

        print(f"\nデータ情報:")
        print(f"  カラム数: {len(df.columns)}")
        print(f"  カラム名: {list(df.columns)}")
        print(f"  データサイズ: {df.shape}")
        print(f"\n最初の5行:")
        print(df.head())

    except Exception as e:
        print(f"エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_utf16_csv()