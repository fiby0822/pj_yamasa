#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1: S3から4つのExcelファイルを取得し、統合してparquetで保存
"""

import pandas as pd
import numpy as np
import boto3
import os
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

def get_s3_client():
    """S3クライアントを取得"""
    return boto3.client('s3',
                       aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                       aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                       region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

def parse_excel_date(series: pd.Series) -> pd.Series:
    """文字/日付/Excelシリアル混在を安全に datetime へ"""
    s = series.copy()
    # 通常の日付形式として解釈
    dt1 = pd.to_datetime(s, errors="coerce")
    # 数値として解釈（Excelシリアル値の可能性）
    num = pd.to_numeric(s, errors="coerce")
    dt2 = pd.to_datetime(num, unit="d", origin="1899-12-30", errors="coerce")
    # 最初の解釈を優先、失敗したら2番目を使用
    return dt1.combine_first(dt2)

def usage_from_name(filename: str) -> str:
    """ファイル名からusage_typeを判定"""
    base = os.path.basename(filename)
    if "家庭用" in base:
        return "household"
    if "業務用" in base:
        return "business"
    return None

def read_excel_from_s3(s3_client, bucket_name, file_key, jp_cols):
    """S3からExcelファイルを読み込む"""
    try:
        print(f"  読込中: s3://{bucket_name}/{file_key}")
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_excel(BytesIO(obj['Body'].read()), usecols=jp_cols, engine="openpyxl")
        print(f"    ✓ 成功: {len(df):,} 行")
        return df
    except Exception as e:
        print(f"    ✗ エラー: {e}")
        return None

def main():
    """メイン処理"""
    print("="*70)
    print("Step 1: S3からExcelファイルを取得し、統合してparquetで保存")
    print("="*70)

    # S3設定
    bucket_name = "fiby-yamasa-prediction"
    s3 = get_s3_client()

    # 対象ファイル
    expected_files = {
        "家庭用202101-202312得意先別出荷.xlsx": "家庭用202101-202312",
        "家庭用202401-202506得意先別出荷.xlsx": "家庭用202401-202506",
        "業務用202101-202312得意先別出荷.xlsx": "業務用202101-202312",
        "業務用202401-202506得意先別出荷.xlsx": "業務用202401-202506",
    }

    # 読み込むカラム
    jp_cols = [
        "出荷日", "品番", "品名", "店番", "出荷数", "曜日（月=1）", "週番号",
        "品目階層1", "品目階層2", "品目階層3", "容量", "容器"
    ]

    # カラム名のマッピング
    rename_map = {
        "出荷日": "file_date",
        "品番": "product_key",
        "品名": "product_name",
        "店番": "store_code",
        "出荷数": "actual_value",
        "曜日（月=1）": "day_of_week_mon1",
        "週番号": "week_number",
        "品目階層1": "category_lvl1",
        "品目階層2": "category_lvl2",
        "品目階層3": "category_lvl3",
        "容量": "volume",
        "容器": "container",
    }

    # ===== 1. ファイルを読み込み =====
    print("\n1. Excelファイルを読み込み中...")
    df_list = []

    for file_key, short_name in expected_files.items():
        df = read_excel_from_s3(s3, bucket_name, file_key, jp_cols)

        if df is not None:
            # ファイル名とusage_typeを追加
            df["file_name"] = file_key
            df["usage_type"] = usage_from_name(file_key)

            # カラム名を英語に変換
            df = df.rename(columns=rename_map)

            # 日付処理
            df["file_date"] = parse_excel_date(df["file_date"])

            # 数値変換
            df["actual_value"] = pd.to_numeric(df["actual_value"], errors="coerce")

            df_list.append(df)
        else:
            print(f"    警告: {file_key} の読み込みに失敗しました")

    if not df_list:
        print("\nエラー: 読み込めたファイルがありません")
        return

    # ===== 2. データを統合 =====
    print("\n2. データを統合中...")
    df = pd.concat(df_list, ignore_index=True)

    # material_keyを作成
    df["material_key"] = df["product_key"] + "_" + df["store_code"]

    print(f"  統合後のサイズ: {df.shape[0]:,} 行 × {df.shape[1]} 列")

    # ===== 3. データ情報を表示 =====
    print("\n3. データ情報:")
    print(f"  期間: {df['file_date'].min()} ～ {df['file_date'].max()}")

    # usage_type毎の統計
    print("\n  usage_type別の統計:")
    usage_stats = df.groupby('usage_type').agg({
        'material_key': lambda x: x.nunique(),
        'file_date': 'count'
    }).rename(columns={'material_key': 'unique_material_keys', 'file_date': 'records'})

    total_keys = df['material_key'].nunique()
    total_records = len(df)

    print("  " + "-"*50)
    for usage_type, row in usage_stats.iterrows():
        keys = row['unique_material_keys']
        records = row['records']
        print(f"  {usage_type}:")
        print(f"    - レコード数: {records:,} ({records/total_records*100:.1f}%)")
        print(f"    - ユニークmaterial_key数: {keys:,} ({keys/total_keys*100:.1f}%)")

    print("  " + "-"*50)
    print(f"  合計:")
    print(f"    - レコード数: {total_records:,}")
    print(f"    - ユニークmaterial_key数: {total_keys:,}")

    # ===== 4. S3に保存（Parquet形式） =====
    print("\n4. Parquet形式で保存中...")

    # データ型の最適化
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            if df[col].min() >= -32768 and df[col].max() <= 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')

    # Parquetバッファに書き込み
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False, compression='snappy')
    parquet_buffer.seek(0)

    # S3に保存
    output_key = "data/df_confirmed_order_input_yamasa.parquet"
    s3.put_object(Bucket=bucket_name, Key=output_key, Body=parquet_buffer.getvalue())

    print(f"  ✓ 保存完了: s3://{bucket_name}/{output_key}")

    print("\n" + "="*70)
    print("✅ Step 1 完了!")
    print("="*70)

    return df

if __name__ == "__main__":
    df = main()