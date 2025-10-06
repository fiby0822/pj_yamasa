#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: 日付補完とゼロ埋め処理
Step 1で作成したparquetファイルを読み込み、欠損日をゼロで埋める
"""

import pandas as pd
import numpy as np
import boto3
import os
from io import BytesIO
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

# ==== 設定 ====
KEY_COL = "material_key"
DATE_COL = "file_date"
TARGET_COL = "actual_value"

# 補完対象（存在する列だけが使われます）
POSSIBLE_COLS_TO_PROPAGATE = [
    "product_key", "product_name", "store_code", "usage_type",
    "category_lvl1", "category_lvl2", "category_lvl3", "container", "file_name",
]

# 圧縮候補（最後に category 化したい列）
POSSIBLE_COLS_TO_CATEGORIZE = [
    "usage_type", "product_key", "product_name", "store_code",
    "file_name", "category_lvl1", "category_lvl2", "category_lvl3", "container"
]

def get_s3_client():
    """S3クライアントを取得"""
    return boto3.client('s3',
                       aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                       aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                       region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

def memory_megabytes(df: pd.DataFrame) -> float:
    """DataFrameのメモリ使用量をMB単位で返す"""
    return float(df.memory_usage(deep=True).sum()) / (1024**2)

def optimize_types(df: pd.DataFrame,
                  to_categorical_cols: list = None,
                  verbose: bool = True) -> pd.DataFrame:
    """データ型を最適化してメモリを削減"""
    before = memory_megabytes(df)

    # 数値縮小
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        if df[c].min() >= -32768 and df[c].max() <= 32767:
            df[c] = df[c].astype("int16")
        else:
            df[c] = df[c].astype("int32")

    # 指定列のみ category 化
    if to_categorical_cols:
        for c in to_categorical_cols:
            if c in df.columns and df[c].dtype == "object":
                df[c] = df[c].astype("category")

    after = memory_megabytes(df)
    if verbose and before > 0:
        reduction = (1 - after/before) * 100
        print(f"  [最適化] {before:,.1f} MB → {after:,.1f} MB ({reduction:,.1f}% 削減)")

    return df

def main():
    """メイン処理"""
    print("="*70)
    print("Step 2: 日付補完とゼロ埋め処理")
    print("="*70)

    # S3設定
    bucket_name = "fiby-yamasa-prediction"
    s3 = get_s3_client()

    # ===== 1. Step 1の出力を読み込み =====
    print("\n1. Step 1の出力データを読み込み中...")
    input_key = "data/df_confirmed_order_input_yamasa.parquet"

    try:
        # S3から読み込み
        obj = s3.get_object(Bucket=bucket_name, Key=input_key)
        df = pd.read_parquet(BytesIO(obj['Body'].read()))
        print(f"  ✓ 読込成功: s3://{bucket_name}/{input_key}")
        print(f"  サイズ: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    except:
        # ローカルから読み込み（フォールバック）
        local_path = "data/df_confirmed_order_input_yamasa.parquet"
        if os.path.exists(local_path):
            df = pd.read_parquet(local_path)
            print(f"  ✓ ローカル読込成功: {local_path}")
            print(f"  サイズ: {df.shape[0]:,} 行 × {df.shape[1]} 列")
        else:
            print("  ✗ エラー: データファイルが見つかりません")
            print("  先に prepare_yamasa_data_step1.py を実行してください")
            return

    # ===== 2. 前処理 =====
    print("\n2. データ前処理中...")

    # 型を安全に整える
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # material_key は一貫して文字列で扱う
    df[KEY_COL] = df[KEY_COL].astype(str)

    # 必須列の欠損除外 & inf→NaN
    df = df.dropna(subset=[KEY_COL, DATE_COL]).replace([np.inf, -np.inf], np.nan).copy()

    print(f"  前処理後: {df.shape[0]:,} 行")

    # ===== 3. (material_key, file_date) で一意化 =====
    print("\n3. データを一意化中...")

    # その日の数量合計
    sum_df = (
        df.groupby([KEY_COL, DATE_COL], as_index=False, sort=False)[TARGET_COL]
        .sum()
    )

    # その日の「最後の行」の属性を代表値に
    last_df = (
        df.sort_values([KEY_COL, DATE_COL], kind="mergesort")
        .groupby([KEY_COL, DATE_COL], as_index=False)
        .last()
    )
    last_df = last_df.drop(columns=[TARGET_COL], errors="ignore")

    # 合体してユニーク化
    df_agg = sum_df.merge(last_df, on=[KEY_COL, DATE_COL], how="left")

    # material_key は引き続き str
    df_agg[KEY_COL] = df_agg[KEY_COL].astype(str)

    # 数値最適化
    df_agg = optimize_types(df_agg, to_categorical_cols=None, verbose=True)

    print(f"  一意化後: {df_agg.shape[0]:,} 行")

    # 補完対象列
    cols_to_propagate = [c for c in POSSIBLE_COLS_TO_PROPAGATE if c in df_agg.columns]

    # 全キー一覧
    all_keys = df_agg[KEY_COL].drop_duplicates().tolist()
    print(f"  ユニークなmaterial_key数: {len(all_keys):,}")

    # usage_type毎の統計を表示
    print("\n  usage_type別のmaterial_key数（補完前）:")
    if 'usage_type' in df_agg.columns:
        usage_stats = df_agg.groupby('usage_type')[KEY_COL].nunique()
        total_keys_pre = df_agg[KEY_COL].nunique()
        for usage, count in usage_stats.items():
            print(f"    {usage}: {count:,} material_keys ({count/total_keys_pre*100:.1f}%)")
        print(f"    合計: {total_keys_pre:,} material_keys")

    # ===== 4. 日付補完とゼロ埋め =====
    print("\n4. 日付補完とゼロ埋め処理中...")

    # 結果を格納するリスト
    all_results = []

    # バッチサイズ
    BATCH = 100  # メモリに合わせて調整（メモリ不足のため小さく）

    for i in tqdm(range(0, len(all_keys), BATCH), desc="  処理進捗"):
        keys = all_keys[i:i+BATCH]
        sub = df_agg[df_agg[KEY_COL].isin(keys)].copy()

        # --- A) 各キーの最小~最大の「日次」カレンダー作成 ---
        dense_parts = []
        for k, g in sub.groupby(KEY_COL, sort=False):
            dmin, dmax = g[DATE_COL].min(), g[DATE_COL].max()
            if pd.isna(dmin) or pd.isna(dmax):
                continue
            idx = pd.date_range(dmin, dmax, freq="D")
            if len(idx) == 0:
                continue
            # material_key は str のまま埋める
            dense_parts.append(pd.DataFrame({KEY_COL: str(k), DATE_COL: idx}))

        if not dense_parts:
            continue

        dense = pd.concat(dense_parts, ignore_index=True)
        dense = dense.sort_values([KEY_COL, DATE_COL], kind="mergesort").reset_index(drop=True)

        # --- B) 値を突合 → 欠損日の actual_value を0に ---
        merged = dense.merge(
            sub[[KEY_COL, DATE_COL, TARGET_COL] + cols_to_propagate],
            on=[KEY_COL, DATE_COL], how="left"
        )
        merged[TARGET_COL] = merged[TARGET_COL].fillna(0).astype("float32")

        # --- C) 属性は「直近の actual_value>0 の行」から前方補完 ---
        if cols_to_propagate:
            mask_pos = merged[TARGET_COL] > 0
            gkey = merged[KEY_COL]  # str

            for c in cols_to_propagate:
                if c not in merged.columns:
                    continue
                src = merged[c].where(mask_pos)  # 正例位置のみ値を残す
                filled = src.groupby(gkey).transform("ffill")  # 前方補完
                merged[c] = merged[c].fillna(filled)  # 欠損だけ埋める

        # --- D) 曜日・週番号・月を再計算 ---
        merged[DATE_COL] = pd.to_datetime(merged[DATE_COL])
        merged["day_of_week_mon1"] = (merged[DATE_COL].dt.dayofweek).astype("int8") + 1
        merged["week_number"] = merged[DATE_COL].dt.isocalendar().week.astype("int16")
        merged["month"] = merged[DATE_COL].dt.month.astype("int8")

        # 数値縮小＋カテゴリ圧縮
        merged = optimize_types(
            merged,
            to_categorical_cols=[c for c in POSSIBLE_COLS_TO_CATEGORIZE if c in merged.columns],
            verbose=False
        )

        all_results.append(merged)

    # ===== 5. 結果を結合 =====
    print("\n5. 結果を結合中...")
    df_final = pd.concat(all_results, ignore_index=True)
    df_final = df_final.sort_values([KEY_COL, DATE_COL]).reset_index(drop=True)

    print(f"  最終サイズ: {df_final.shape[0]:,} 行 × {df_final.shape[1]} 列")

    # usage_type毎の統計を表示
    print("\n  usage_type別のmaterial_key数（補完後）:")
    if 'usage_type' in df_final.columns:
        usage_stats_final = df_final.groupby('usage_type')[KEY_COL].nunique()
        total_keys_final = df_final[KEY_COL].nunique()
        for usage, count in usage_stats_final.items():
            print(f"    {usage}: {count:,} material_keys ({count/total_keys_final*100:.1f}%)")
        print(f"    合計: {total_keys_final:,} material_keys")

        # レコード数も表示
        print("\n  usage_type別のレコード数（補完後）:")
        record_stats = df_final['usage_type'].value_counts()
        total_records = len(df_final)
        for usage, count in record_stats.items():
            print(f"    {usage}: {count:,} レコード ({count/total_records*100:.1f}%)")

    # ===== 6. 保存 =====
    print("\n6. 結果を保存中...")

    # Parquet形式で保存
    parquet_buffer = BytesIO()
    df_final.to_parquet(parquet_buffer, index=False, compression='snappy')
    parquet_buffer.seek(0)

    output_key = "data/df_confirmed_order_input_yamasa_fill_zero.parquet"
    s3.put_object(Bucket=bucket_name, Key=output_key, Body=parquet_buffer.getvalue())
    print(f"  ✓ Parquet保存: s3://{bucket_name}/{output_key}")

    # S3 CSVとローカル保存は行わない（ユーザー指示により削除）

    # ===== 7. 最終統計 =====
    print("\n7. 最終統計:")
    print(f"  期間: {df_final[DATE_COL].min()} ～ {df_final[DATE_COL].max()}")
    print(f"  ゼロの数: {(df_final[TARGET_COL] == 0).sum():,} ({(df_final[TARGET_COL] == 0).sum() / len(df_final) * 100:.1f}%)")
    print(f"  非ゼロの数: {(df_final[TARGET_COL] > 0).sum():,} ({(df_final[TARGET_COL] > 0).sum() / len(df_final) * 100:.1f}%)")
    print(f"  メモリ使用量: {memory_megabytes(df_final):,.1f} MB")

    print("\n" + "="*70)
    print("✅ Step 2 完了!")
    print("="*70)

    return df_final

if __name__ == "__main__":
    df = main()