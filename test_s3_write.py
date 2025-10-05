#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S3書き込み権限テスト"""

import boto3
import os
from datetime import datetime
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

def test_s3_write():
    """S3への書き込み権限をテスト"""

    bucket_name = "fiby-yamasa-prediction"
    test_key = f"output_data/test/test_write_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))

    print(f"バケット: {bucket_name}")
    print(f"テストファイル: {test_key}")

    try:
        # テストファイルを書き込み
        test_content = f"Test write at {datetime.now()}"
        s3.put_object(
            Bucket=bucket_name,
            Key=test_key,
            Body=test_content.encode('utf-8')
        )
        print("✓ S3への書き込み成功！")

        # 削除も試す
        s3.delete_object(Bucket=bucket_name, Key=test_key)
        print("✓ S3からの削除も成功！")

        return True

    except Exception as e:
        print(f"✗ S3への書き込みエラー: {e}")

        # 別の場所を試す
        test_key2 = f"test_write_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        print(f"\n別の場所を試します: {test_key2}")

        try:
            s3.put_object(
                Bucket=bucket_name,
                Key=test_key2,
                Body=test_content.encode('utf-8')
            )
            print("✓ ルートディレクトリへの書き込み成功！")
            s3.delete_object(Bucket=bucket_name, Key=test_key2)
            return True
        except Exception as e2:
            print(f"✗ ルートディレクトリへの書き込みもエラー: {e2}")
            return False

if __name__ == "__main__":
    test_s3_write()