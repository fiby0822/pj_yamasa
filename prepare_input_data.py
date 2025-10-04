#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Excelファイルから入力データを準備するスクリプト
S3上のExcelファイルを読み込み、CSVに変換して保存
"""

import pandas as pd
import boto3
import os
from io import BytesIO
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

def read_excel_from_s3(bucket_name, file_key):
    """S3からExcelファイルを読み込む"""
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_excel(BytesIO(obj['Body'].read()), sheet_name=None)
        print(f"Excel読込成功: s3://{bucket_name}/{file_key}")
        return df
    except Exception as e:
        print(f"Excel読込エラー: {e}")
        return None

def save_csv_to_s3(df, bucket_name, file_key):
    """DataFrameをCSVとしてS3に保存"""
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))
    try:
        csv_buffer = df.to_csv(index=False)
        s3.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer)
        print(f"CSV保存成功: s3://{bucket_name}/{file_key}")
        return True
    except Exception as e:
        print(f"CSV保存エラー: {e}")
        return False

def prepare_input_data():
    """Excelファイルを結合して入力データを準備"""
    bucket_name = "fiby-yamasa-prediction"

    # Excelファイルのリスト
    excel_files = [
        "家庭用202101-202312得意先別出荷.xlsx",
        "家庭用202401-202506得意先別出荷.xlsx",
        "業務用202101-202312得意先別出荷.xlsx",
        "業務用202401-202506得意先別出荷.xlsx",
        "202101-202506大口得意先別出荷.xlsx"
    ]

    all_data = []

    print("=" * 50)
    print("入力データ準備処理開始")
    print("=" * 50)

    # 各Excelファイルを読み込み
    for file_key in excel_files:
        print(f"\n処理中: {file_key}")
        excel_data = read_excel_from_s3(bucket_name, file_key)

        if excel_data is not None:
            # 各シートを処理
            for sheet_name, df in excel_data.items():
                print(f"  シート '{sheet_name}': {df.shape}")

                # 必要なカラムの標準化（実際のカラム名に応じて調整が必要）
                # ここは実際のデータ構造に基づいて修正が必要です
                df_processed = df.copy()

                # ファイルタイプを追加
                if "家庭用" in file_key:
                    df_processed['category'] = '家庭用'
                elif "業務用" in file_key:
                    df_processed['category'] = '業務用'
                else:
                    df_processed['category'] = '大口'

                all_data.append(df_processed)

    if all_data:
        # データを結合
        df_combined = pd.concat(all_data, ignore_index=True)
        print(f"\n結合後のデータサイズ: {df_combined.shape}")

        # 必要な前処理を実行（例：日付変換、欠損値補完など）
        # ここは実際のデータ構造に基づいて修正が必要です

        # material_keyカラムを作成（仮の実装）
        if 'product_code' in df_combined.columns:
            df_combined['material_key'] = df_combined['product_code']
        elif 'item_code' in df_combined.columns:
            df_combined['material_key'] = df_combined['item_code']
        else:
            # 最初のカラムをmaterial_keyとして使用
            df_combined['material_key'] = df_combined.iloc[:, 0]

        # file_dateカラムを作成（仮の実装）
        if 'date' in df_combined.columns:
            df_combined['file_date'] = pd.to_datetime(df_combined['date'])
        elif 'ship_date' in df_combined.columns:
            df_combined['file_date'] = pd.to_datetime(df_combined['ship_date'])
        else:
            # デフォルトの日付を設定
            df_combined['file_date'] = pd.to_datetime('2024-01-01')

        # actual_valueカラムを作成（仮の実装）
        numeric_columns = df_combined.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) > 0:
            # 数値カラムの最初のものを使用
            df_combined['actual_value'] = df_combined[numeric_columns[0]]
        else:
            df_combined['actual_value'] = 0

        # ゼロ埋め処理
        df_combined['actual_value'] = df_combined['actual_value'].fillna(0)

        # CSVとして保存
        output_key = "df_confirmed_order_input_yamasa_fill_zero.csv"
        success = save_csv_to_s3(df_combined, bucket_name, output_key)

        if success:
            print("\n" + "=" * 50)
            print("入力データ準備完了")
            print("=" * 50)
            print(f"\n作成されたファイル: s3://{bucket_name}/{output_key}")
            print(f"レコード数: {len(df_combined)}")
            print(f"カラム: {list(df_combined.columns)}")

            return df_combined
        else:
            print("\n入力データの保存に失敗しました")
            return None
    else:
        print("\nデータが見つかりませんでした")
        return None

if __name__ == "__main__":
    df = prepare_input_data()
    if df is not None:
        print("\n最初の5行:")
        print(df.head())