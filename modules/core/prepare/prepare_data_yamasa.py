#!/usr/bin/env python3
"""
S3からヤマサの出荷データ（Excel）を読み込み、加工・統合してParquet形式で保存
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from data_io.s3_handler import S3Handler

# 期待するファイル名
EXPECTED_FILES = {
    "家庭用202101-202312得意先別出荷": "household",
    "家庭用202401-202506得意先別出荷": "household",
    "業務用202401-202506得意先別出荷": "business",
    "業務用202101-202312得意先別出荷": "business",
}

# カラム設定
JP_COLS = [
    "出荷日", "品番", "品名", "店番", "出荷数", "曜日（月=1）", "週番号",
    "品目階層1", "品目階層2", "品目階層3", "容量", "容器"
]

RENAME_MAP = {
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

def parse_excel_date(series: pd.Series) -> pd.Series:
    """文字/日付/Excelシリアル混在を安全にdatetimeへ変換"""
    s = series.copy()
    # 直接datetime変換を試みる
    dt1 = pd.to_datetime(s, errors="coerce")
    # 数値としてパースしてExcelシリアル日付として変換
    num = pd.to_numeric(s, errors="coerce")
    dt2 = pd.to_datetime(num, unit="d", origin="1899-12-30", errors="coerce")
    # 最初の変換結果を優先、失敗したら2番目の結果を使用
    return dt1.combine_first(dt2)

def main():
    print("=== prepare_data_yamasa.py 開始 ===")

    # S3ハンドラー初期化
    s3 = S3Handler()

    # S3上のExcelファイルをリスト
    input_files = s3.list_files("input_data/")
    print(f"S3 input_data内のファイル数: {len(input_files)}")

    # 必要なファイルを探す
    target_files = {}
    for file_key in input_files:
        file_name = os.path.basename(file_key)
        base_name = os.path.splitext(file_name)[0]
        if base_name in EXPECTED_FILES and file_key.endswith(".xlsx"):
            target_files[base_name] = file_key
            print(f"対象ファイル発見: {file_name}")

    missing = [k for k in EXPECTED_FILES.keys() if k not in target_files]
    if missing:
        print(f"[警告] 見つからないファイル: {missing}")

    if not target_files:
        raise FileNotFoundError("必要なExcelファイルがS3に見つかりませんでした")

    # 各ファイルを読み込んで処理
    df_list = []
    for base_name, file_key in target_files.items():
        print(f"\n処理中: {base_name}")

        # Excelファイル読み込み
        df_temp = s3.read_excel(file_key, usecols=JP_COLS, engine="openpyxl")

        # ファイル名とusage_type追加
        df_temp["file_name"] = os.path.basename(file_key)
        df_temp["usage_type"] = EXPECTED_FILES[base_name]

        # カラム名変更
        df_temp = df_temp.rename(columns=RENAME_MAP)

        # 日付変換
        df_temp["file_date"] = parse_excel_date(df_temp["file_date"])

        # actual_value数値化
        df_temp["actual_value"] = pd.to_numeric(df_temp["actual_value"], errors="coerce")

        print(f"  - レコード数: {len(df_temp):,}")
        print(f"  - 期間: {df_temp['file_date'].min()} ～ {df_temp['file_date'].max()}")

        df_list.append(df_temp)

    # 全データ結合
    df = pd.concat(df_list, ignore_index=True)

    # material_key作成
    df["material_key"] = df["product_key"].astype(str) + "_" + df["store_code"].astype(str)

    # 結果確認
    print(f"\n=== 統合結果 ===")
    print(f"総レコード数: {len(df):,}")
    print(f"期間: {df['file_date'].min()} ～ {df['file_date'].max()}")
    print(f"usage_type別レコード数:")
    print(df['usage_type'].value_counts())
    print(f"material_key数: {df['material_key'].nunique():,}")

    # S3に保存
    output_key = "output/df_confirmed_order_input_yamasa.parquet"
    s3.write_parquet(df, output_key)

    print(f"\n=== 完了 ===")
    print(f"保存先: s3://fiby-yamasa-prediction/{output_key}")
    print(f"データ shape: {df.shape}")

if __name__ == "__main__":
    main()