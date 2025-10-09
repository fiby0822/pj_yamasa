import os
import boto3
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from io import BytesIO
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class S3Handler:
    def __init__(self, bucket_name: str = "fiby-yamasa-prediction"):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1')
        )

    def list_files(self, prefix: str) -> list:
        """S3のファイル一覧を取得"""
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix
        )
        if 'Contents' not in response:
            return []
        return [obj['Key'] for obj in response['Contents']]

    def read_excel(self, key: str, **kwargs) -> pd.DataFrame:
        """S3からExcelファイルを読み込み"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            content = response['Body'].read()
            return pd.read_excel(BytesIO(content), **kwargs)
        except Exception as e:
            print(f"Error reading Excel file {key}: {str(e)}")
            raise

    def read_parquet(self, key: str) -> pd.DataFrame:
        """S3からParquetファイルを読み込み"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            content = response['Body'].read()
            return pd.read_parquet(BytesIO(content))
        except Exception as e:
            print(f"Error reading Parquet file {key}: {str(e)}")
            raise

    def write_parquet(self, df: pd.DataFrame, key: str, compression: str = 'snappy') -> None:
        """DataFrameをParquet形式でS3に保存"""
        try:
            buffer = BytesIO()
            table = pa.Table.from_pandas(df)
            pq.write_table(table, buffer, compression=compression)
            buffer.seek(0)

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.getvalue()
            )
            print(f"Successfully saved to s3://{self.bucket_name}/{key}")
        except Exception as e:
            print(f"Error writing Parquet file {key}: {str(e)}")
            raise

    def write_csv(self, df: pd.DataFrame, key: str, **kwargs) -> None:
        """DataFrameをCSV形式でS3に保存"""
        try:
            buffer = BytesIO()
            df.to_csv(buffer, index=False, **kwargs)
            buffer.seek(0)

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.getvalue()
            )
            print(f"Successfully saved CSV to s3://{self.bucket_name}/{key}")
        except Exception as e:
            print(f"Error writing CSV file {key}: {str(e)}")
            raise

    def file_exists(self, key: str) -> bool:
        """S3上のファイルの存在確認"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except:
            return False

    def download_file(self, key: str, local_path: str) -> None:
        """S3からファイルをローカルにダウンロード"""
        try:
            self.s3_client.download_file(self.bucket_name, key, local_path)
            print(f"Downloaded s3://{self.bucket_name}/{key} to {local_path}")
        except Exception as e:
            print(f"Error downloading file {key}: {str(e)}")
            raise