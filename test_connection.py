#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S3接続テストスクリプト
"""

import boto3
import os
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

def test_s3_connection():
    """S3接続をテスト"""
    try:
        # S3クライアント作成
        s3 = boto3.client('s3',
                          aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                          aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                          region_name=os.getenv('AWS_DEFAULT_REGION'))

        bucket_name = "fiby-yamasa-prediction"

        print("S3接続テスト開始")
        print(f"バケット: {bucket_name}")
        print(f"リージョン: {os.getenv('AWS_DEFAULT_REGION')}")

        # バケット内のオブジェクトをリスト
        response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=10)

        if 'Contents' in response:
            print(f"\nバケット内のファイル（最初の10個）:")
            for obj in response['Contents']:
                print(f"  - {obj['Key']} ({obj['Size']/1024/1024:.1f} MB)")
        else:
            print("\nバケットは空です")

        # 特定のファイルの存在確認
        target_file = "df_confirmed_order_input_yamasa_fill_zero.csv"
        try:
            s3.head_object(Bucket=bucket_name, Key=target_file)
            print(f"\n✓ 入力ファイル '{target_file}' が存在します")
        except:
            print(f"\n✗ 入力ファイル '{target_file}' が見つかりません")

        print("\nS3接続テスト成功!")
        return True

    except Exception as e:
        print(f"\nS3接続エラー: {e}")
        return False

if __name__ == "__main__":
    test_s3_connection()