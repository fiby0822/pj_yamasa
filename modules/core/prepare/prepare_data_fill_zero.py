#!/usr/bin/env python3
"""
実績値ゼロのレコードを補完してS3に保存
material_key毎に期間内の全日次データを生成し、欠損日はactual_value=0として補完
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from data_io.s3_handler import S3Handler

# 設定
KEY_COL = "material_key"
DATE_COL = "file_date"
TARGET_COL = "actual_value"

# 補完対象列（存在する列のみ使用、曜日・週番号は再計算）
POSSIBLE_COLS_TO_PROPAGATE = [
    "product_key", "product_name", "store_code", "usage_type",
    "category_lvl1", "category_lvl2", "category_lvl3", "container_f", "file_name",
]

# カテゴリ変換対象列（メモリ最適化）
POSSIBLE_COLS_TO_CATEGORIZE = [
    "usage_type", "product_key", "product_name", "store_code",
    "file_name", "category_lvl1", "category_lvl2", "category_lvl3", "container_f"
]

def memory_megabytes(df: pd.DataFrame) -> float:
    """DataFrameのメモリ使用量をMB単位で取得"""
    return float(df.memory_usage(deep=True).sum()) / (1024**2)

def optimize_types(df: pd.DataFrame,
                  to_categorical_cols: list = None,
                  verbose: bool = True) -> pd.DataFrame:
    """DataFrameの型を最適化してメモリ使用量を削減"""
    before = memory_megabytes(df)

    # 数値型の縮小
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        if c in ["day_of_week_mon1_f"]:
            df[c] = df[c].astype("int8")
        elif c in ["week_number_f"]:
            df[c] = df[c].astype("int16")
        else:
            df[c] = df[c].astype("int32")

    # カテゴリ型への変換
    if to_categorical_cols:
        for c in to_categorical_cols:
            if c in df.columns and df[c].dtype == "object":
                df[c] = df[c].astype("category")

    after = memory_megabytes(df)
    if verbose and before > 0:
        reduction_pct = (1 - after/before) * 100
        print(f"[メモリ最適化] {before:,.1f} MB → {after:,.1f} MB ({reduction_pct:,.1f}% 削減)")

    return df

def process_batch(df_batch: pd.DataFrame, cols_to_propagate: list) -> pd.DataFrame:
    """バッチ単位でゼロ値補完処理を実行"""
    dense_parts = []

    # 各material_keyについて日次カレンダー作成
    for key, group in df_batch.groupby(KEY_COL, sort=False):
        date_min = group[DATE_COL].min()
        date_max = group[DATE_COL].max()

        if pd.isna(date_min) or pd.isna(date_max):
            continue

        # 日次インデックス生成
        date_range = pd.date_range(date_min, date_max, freq="D")
        if len(date_range) == 0:
            continue

        # material_keyは文字列として保持
        dense_df = pd.DataFrame({
            KEY_COL: str(key),
            DATE_COL: date_range
        })
        dense_parts.append(dense_df)

    if not dense_parts:
        return pd.DataFrame()

    # 全キーの密なデータフレーム作成
    dense = pd.concat(dense_parts, ignore_index=True)
    dense = dense.sort_values([KEY_COL, DATE_COL], kind="mergesort").reset_index(drop=True)

    # 元データとマージ
    merged = dense.merge(
        df_batch[[KEY_COL, DATE_COL, TARGET_COL] + cols_to_propagate],
        on=[KEY_COL, DATE_COL],
        how="left"
    )

    # actual_valueの欠損値を0で埋める
    merged[TARGET_COL] = merged[TARGET_COL].fillna(0).astype("float32")

    # 属性列の前方補完（actual_value > 0の行から）
    if cols_to_propagate:
        mask_positive = merged[TARGET_COL] > 0
        group_key = merged[KEY_COL]

        for col in cols_to_propagate:
            if col not in merged.columns:
                continue

            # 正値位置のみ値を残す
            source = merged[col].where(mask_positive)
            # グループごとに前方補完
            filled = source.groupby(group_key, observed=False).transform("ffill")
            # 欠損値のみ埋める
            merged[col] = merged[col].fillna(filled)

    # 曜日・週番号・月を再計算
    merged[DATE_COL] = pd.to_datetime(merged[DATE_COL])
    merged["day_of_week_mon1_f"] = (merged[DATE_COL].dt.dayofweek + 1).astype("int8")
    merged["week_number_f"] = merged[DATE_COL].dt.isocalendar().week.astype("int16")
    merged["month"] = merged[DATE_COL].dt.to_period("M").dt.to_timestamp()

    return merged

def main():
    print("=== prepare_data_fill_zero.py 開始 ===")

    # S3ハンドラー初期化
    s3 = S3Handler()

    # 入力データ読み込み
    input_key = "output/df_confirmed_order_input_yamasa.parquet"
    print(f"入力データ読み込み中: s3://fiby-yamasa-prediction/{input_key}")
    df = s3.read_parquet(input_key)

    print(f"読み込み完了 - レコード数: {len(df):,}")

    # データ前処理
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df[KEY_COL] = df[KEY_COL].astype(str)

    # 欠損値とinf値の処理
    df = df.dropna(subset=[KEY_COL, DATE_COL]).replace([np.inf, -np.inf], np.nan).copy()

    # (material_key, file_date)で一意化
    # 同日の数量を合計
    sum_df = (
        df.groupby([KEY_COL, DATE_COL], as_index=False, sort=False)[TARGET_COL]
        .sum()
    )

    # 同日の最後の行の属性を代表値として使用
    last_df = (
        df.sort_values([KEY_COL, DATE_COL], kind="mergesort")
        .groupby([KEY_COL, DATE_COL], as_index=False)
        .last()
    )
    last_df = last_df.drop(columns=[TARGET_COL], errors="ignore")

    # マージして一意化
    df_agg = sum_df.merge(last_df, on=[KEY_COL, DATE_COL], how="left")
    df_agg[KEY_COL] = df_agg[KEY_COL].astype(str)

    # メモリ最適化（数値型のみ）
    df_agg = optimize_types(df_agg, to_categorical_cols=None, verbose=True)

    # 補完対象列の特定
    cols_to_propagate = [c for c in POSSIBLE_COLS_TO_PROPAGATE if c in df_agg.columns]
    print(f"補完対象列: {cols_to_propagate}")

    # 全material_keyのリスト取得
    all_keys = df_agg[KEY_COL].drop_duplicates().tolist()
    print(f"material_key総数: {len(all_keys):,}")

    # バッチ処理
    BATCH_SIZE = 800  # メモリに応じて調整
    result_dfs = []

    print("\nバッチ処理開始...")
    for i in tqdm(range(0, len(all_keys), BATCH_SIZE), desc="補完処理"):
        batch_keys = all_keys[i:i+BATCH_SIZE]
        batch_df = df_agg[df_agg[KEY_COL].isin(batch_keys)].copy()

        # バッチ処理実行
        processed_batch = process_batch(batch_df, cols_to_propagate)

        if not processed_batch.empty:
            # メモリ最適化
            processed_batch = optimize_types(
                processed_batch,
                to_categorical_cols=[c for c in POSSIBLE_COLS_TO_CATEGORIZE if c in processed_batch.columns],
                verbose=False
            )
            result_dfs.append(processed_batch)

    # 全バッチ結果を結合
    print("\nバッチ結果を結合中...")
    df_filled = pd.concat(result_dfs, ignore_index=True)

    # 最終的なデータ確認
    print(f"\n=== 補完処理完了 ===")
    print(f"最終レコード数: {len(df_filled):,}")
    print(f"期間: {df_filled[DATE_COL].min()} ～ {df_filled[DATE_COL].max()}")
    print(f"material_key数: {df_filled[KEY_COL].nunique():,}")
    print(f"ゼロ値レコード数: {(df_filled[TARGET_COL] == 0).sum():,}")
    print(f"非ゼロ値レコード数: {(df_filled[TARGET_COL] > 0).sum():,}")

    # S3に保存
    output_key = "output/df_confirmed_order_input_yamasa_fill_zero.parquet"
    s3.write_parquet(df_filled, output_key)

    print(f"\n=== 完了 ===")
    print(f"保存先: s3://fiby-yamasa-prediction/{output_key}")
    print(f"データ shape: {df_filled.shape}")
    print(f"メモリ使用量: {memory_megabytes(df_filled):,.1f} MB")

if __name__ == "__main__":
    main()