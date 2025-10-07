#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""特徴量作成（曜日別過去平均特徴量を強化版）"""

import pandas as pd
import numpy as np
import boto3
from smart_open import open as smart_open
import os
from datetime import datetime
from dotenv import load_dotenv
import jpholiday

# 環境変数を読み込み
load_dotenv()

# window_size_configの定義（ノートブックと同じ設定）
window_size_config = {
    "material_key": {
        "lag": [1,2,3],
        "rolling_mean": [2,3,4,5,6],
        "rolling_std": [2,3,4,5,6],
        "rate_of_change": [1],
        "cumulative_mean": [2,3,4,5,6,9,12],
    },
    "product_key": {
        "lag": [1,2,3,4,5,6],
        "cumulative_mean": [2,3,4,5,6,9,12],
    },
    "store_code": {
        "lag": [1,2,3,4,5,6],
        "cumulative_mean": [2,3,4,5,6,9,12],
    },
    "category_lvl1": {
        "lag": [1,2,3,4,5,6],
        "cumulative_mean": [2,3,4,5,6,9,12],
    },
    "category_lvl2": {
        "lag": [1,2,3,4,5,6],
        "cumulative_mean": [2,3,4,5,6,9,12],
    },
    "category_lvl3": {
        "lag": [1,2,3,4,5,6],
        "cumulative_mean": [2,3,4,5,6,9,12],
    },
    "overall": {
        "lag": [1,2,3],
        "rolling_mean": [2,3,4,5,6],
        "rolling_std": [2,3,4,5,6],
        "rate_of_change": [1],
        "cumulative_mean": [2,3,4,5,6,9,12],
    },
}

def _add_timeseries_features(df, window_size_config, model_type="confirmed_order_demand_yamasa"):
    """時系列特徴量を追加（曜日別過去平均を強化）"""

    print("="*50)
    print("時系列特徴量作成開始（曜日別過去平均強化版）")
    print("="*50)

    # カラム名のマッピング（大文字を小文字に、スペースをアンダースコアに）
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # actual_valueがない場合は作成
    if 'actual_value' not in df.columns and 'actual value' in df.columns:
        df.rename(columns={'actual value': 'actual_value'}, inplace=True)
    elif 'actual_value' not in df.columns:
        df['actual_value'] = pd.to_numeric(df.get('actual_value', 0), errors='coerce').fillna(0)

    # file_dateの処理
    if 'file_date' not in df.columns and 'file date' in df.columns:
        df.rename(columns={'file date': 'file_date'}, inplace=True)

    # ソート
    if 'file_date' in df.columns:
        df['file_date'] = pd.to_datetime(df['file_date'], errors='coerce')
        df = df.sort_values('file_date')

    # 日付特徴量（_fサフィックス付き）
    if 'file_date' in df.columns:
        print("日付特徴量作成中...")
        df['year_f'] = df['file_date'].dt.year
        df['month_f'] = df['file_date'].dt.month
        df['day_f'] = df['file_date'].dt.day
        df['day_of_week_f'] = df['file_date'].dt.dayofweek
        df['week_of_year_f'] = df['file_date'].dt.isocalendar().week
        df['quarter_f'] = df['file_date'].dt.quarter
        df['is_month_end_f'] = df['file_date'].dt.is_month_end.astype(int)
        df['is_month_start_f'] = df['file_date'].dt.is_month_start.astype(int)

        # 営業日フラグ
        is_weekend = df['file_date'].dt.dayofweek.isin([5, 6])
        is_year_end = ((df['file_date'].dt.month == 12) &
                       (df['file_date'].dt.day.isin([30, 31])))

        def is_japan_holiday(date):
            """jpholidayを使って祝日判定"""
            try:
                if date.month == 1 and date.day in [2, 3]:
                    return True
                return jpholiday.is_holiday(date)
            except:
                return False

        is_holiday = df['file_date'].apply(is_japan_holiday)
        df['is_business_day_f'] = (~(is_weekend | is_year_end | is_holiday)).astype(int)
        df['is_friday_f'] = (df['file_date'].dt.dayofweek == 4).astype(int)
        df['dow_month_interaction_f'] = df['day_of_week_f'] * df['month_f']
        df['is_weekly_milestone_f'] = df['day_f'].isin([7, 14, 21, 28]).astype(int)

    # material_key毎の特徴量作成
    if 'material_key' in df.columns:
        print("material_key毎の特徴量作成中...")
        df = df.sort_values(['material_key', 'file_date'])

        # ラグ特徴量
        for lag in window_size_config["material_key"]["lag"]:
            df[f'lag_{lag}_f'] = df.groupby('material_key')['actual_value'].shift(lag)

        # 移動平均
        for rolling_mean in window_size_config["material_key"]["rolling_mean"]:
            df[f'rolling_mean_{rolling_mean}_f'] = df.groupby('material_key')['actual_value'].shift(1).transform(
                lambda x: x.rolling(window=rolling_mean).mean()
            )

        # 移動標準偏差
        for rolling_std in window_size_config["material_key"]["rolling_std"]:
            df[f'rolling_std_{rolling_std}_f'] = df.groupby('material_key')['actual_value'].shift(1).transform(
                lambda x: x.rolling(window=rolling_std).std()
            )

        # 累積平均
        for cumulative_mean in window_size_config["material_key"]["cumulative_mean"]:
            df[f'cumulative_mean_{cumulative_mean}_f'] = df.groupby('material_key')['actual_value'].transform(
                lambda x: x.rolling(window=cumulative_mean, min_periods=1).mean().shift(1)
            )

        # 週初からの累積平均
        if 'day_of_week_f' in df.columns:
            df['week_start'] = (df['day_of_week_f'] == 0).astype(int).cumsum()
            df['week_to_date_mean_f'] = df.groupby(['material_key', 'week_start'])['actual_value'].transform(
                lambda x: x.expanding().mean().shift(1).fillna(0)
            )
            df = df.drop('week_start', axis=1)

        # Material Key × 曜日の過去平均（既存の特徴量）
        if 'day_of_week_f' in df.columns:
            df['material_dow_mean_f'] = df.groupby(['material_key', 'day_of_week_f'])['actual_value'].transform(
                lambda x: x.expanding().mean().shift(1).fillna(0)
            )

        print("material_key features done")

    # 確定注文予測(ヤマサ)用の特徴量作成
    if model_type == "confirmed_order_demand_yamasa":

        # 品名毎 (product_key)
        if 'product_key' in df.columns:
            print("product_key毎の特徴量作成中...")
            df = df.sort_values(['product_key', 'file_date'])

            for lag in window_size_config["product_key"]["lag"]:
                df[f'product_key_lag_{lag}_f'] = df.groupby('product_key')['actual_value'].shift(lag)

            for cumulative_mean in window_size_config["product_key"]["cumulative_mean"]:
                df[f'product_key_cumulative_mean_{cumulative_mean}_f'] = df.groupby('product_key')['actual_value'].transform(
                    lambda x: x.rolling(window=cumulative_mean, min_periods=1).mean().shift(1)
                )

            # ★新規追加: product_key × 曜日の過去平均
            if 'day_of_week_f' in df.columns:
                print("  - product_key × 曜日の過去平均を追加...")
                df['product_dow_mean_f'] = df.groupby(['product_key', 'day_of_week_f'])['actual_value'].transform(
                    lambda x: x.expanding().mean().shift(1).fillna(0)
                )
                # より最近のデータを重視した加重平均版も追加
                df['product_dow_ewm_f'] = df.groupby(['product_key', 'day_of_week_f'])['actual_value'].transform(
                    lambda x: x.ewm(span=4, adjust=False).mean().shift(1).fillna(0)
                )

            print("product_key features done")

        # 店番毎 (store_code)
        if 'store_code' in df.columns:
            print("store_code毎の特徴量作成中...")
            df = df.sort_values(['store_code', 'file_date'])

            for lag in window_size_config["store_code"]["lag"]:
                df[f'store_code_lag_{lag}_f'] = df.groupby('store_code')['actual_value'].shift(lag)

            for cumulative_mean in window_size_config["store_code"]["cumulative_mean"]:
                df[f'store_code_cumulative_mean_{cumulative_mean}_f'] = df.groupby('store_code')['actual_value'].transform(
                    lambda x: x.rolling(window=cumulative_mean, min_periods=1).mean().shift(1)
                )

            # ★新規追加: store_code × 曜日の過去平均
            if 'day_of_week_f' in df.columns:
                print("  - store_code × 曜日の過去平均を追加...")
                df['store_dow_mean_f'] = df.groupby(['store_code', 'day_of_week_f'])['actual_value'].transform(
                    lambda x: x.expanding().mean().shift(1).fillna(0)
                )
                # より最近のデータを重視した加重平均版も追加
                df['store_dow_ewm_f'] = df.groupby(['store_code', 'day_of_week_f'])['actual_value'].transform(
                    lambda x: x.ewm(span=4, adjust=False).mean().shift(1).fillna(0)
                )

            print("store_code features done")

        # Usage Type毎 (usage_type)の特徴量
        if 'usage_type' in df.columns:
            print("usage_type毎の特徴量作成中...")
            df = df.sort_values(['usage_type', 'file_date'])

            # ラグ特徴量
            for lag in [1, 2, 3, 4, 5, 6]:
                df[f'usage_type_lag_{lag}_f'] = df.groupby('usage_type')['actual_value'].shift(lag)

            # 累積平均
            for cumulative_mean in [2, 3, 4, 5, 6, 9, 12]:
                df[f'usage_type_cumulative_mean_{cumulative_mean}_f'] = df.groupby('usage_type')['actual_value'].transform(
                    lambda x: x.rolling(window=cumulative_mean, min_periods=1).mean().shift(1)
                )

            # ★新規追加: usage_type × 曜日の過去平均
            if 'day_of_week_f' in df.columns:
                print("  - usage_type × 曜日の過去平均を追加...")
                df['usage_dow_mean_f'] = df.groupby(['usage_type', 'day_of_week_f'])['actual_value'].transform(
                    lambda x: x.expanding().mean().shift(1).fillna(0)
                )
                # より最近のデータを重視した加重平均版も追加
                df['usage_dow_ewm_f'] = df.groupby(['usage_type', 'day_of_week_f'])['actual_value'].transform(
                    lambda x: x.ewm(span=4, adjust=False).mean().shift(1).fillna(0)
                )

            print("usage_type features done")

        # カテゴリ毎の特徴量
        for category_col, category_name in zip(['category_lvl1', 'category_lvl2', 'category_lvl3'],
                                               ['category_lvl1', 'category_lvl2', 'category_lvl3']):
            if category_col in df.columns:
                print(f"{category_name}毎の特徴量作成中...")
                df = df.sort_values([category_col, 'file_date'])

                for lag in window_size_config[category_name]["lag"]:
                    df[f'{category_name}_lag_{lag}_f'] = df.groupby(category_col)['actual_value'].shift(lag)

                for cumulative_mean in window_size_config[category_name]["cumulative_mean"]:
                    df[f'{category_name}_cumulative_mean_{cumulative_mean}_f'] = df.groupby(category_col)['actual_value'].transform(
                        lambda x: x.rolling(window=cumulative_mean, min_periods=1).mean().shift(1)
                    )
                print(f"{category_name} features done")

        # 全体(overall)の特徴量
        if 'file_date' in df.columns:
            print("全体の特徴量作成中...")
            df = df.sort_values('file_date')

            # ラグ特徴量
            for lag in window_size_config["overall"]["lag"]:
                df[f'overall_lag_{lag}_f'] = df['actual_value'].shift(lag)

            # 移動平均
            for rolling_mean in window_size_config["overall"]["rolling_mean"]:
                df[f'overall_rolling_mean_{rolling_mean}_f'] = df['actual_value'].shift(1).rolling(window=rolling_mean).mean()

            # 移動標準偏差
            for rolling_std in window_size_config["overall"]["rolling_std"]:
                df[f'overall_rolling_std_{rolling_std}_f'] = df['actual_value'].shift(1).rolling(window=rolling_std).std()

            # 累積平均
            for cumulative_mean in window_size_config["overall"]["cumulative_mean"]:
                df[f'overall_cumulative_mean_{cumulative_mean}_f'] = df['actual_value'].rolling(window=cumulative_mean, min_periods=1).mean().shift(1)

            print("overall features done")

    # 欠損値処理
    print("欠損値処理中...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    print("="*50)
    print("特徴量作成完了")
    print(f"作成された特徴量数: {len(df.columns)}")
    print("="*50)

    return df

def load_data_from_local():
    """ローカルからデータを読み込む"""
    file_path = 'data/df_confirmed_order_input_yamasa_fill_zero.parquet'

    print(f"ローカルデータ読み込み中: {file_path}")

    df = pd.read_parquet(file_path)

    print(f"データ読み込み完了: {len(df)}行")
    return df

def save_features_to_s3(df, suffix="enhanced_dow"):
    """特徴量をS3に保存（README.md構造準拠）"""
    bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', 'fiby-yamasa-prediction')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # README.md構造に従ったパス
    file_key = f'output_data/features/df_features_yamasa_{timestamp}.parquet'
    latest_key = f'output_data/features/df_features_yamasa_latest.parquet'

    s3_path = f's3://{bucket_name}/{file_key}'
    s3_latest_path = f's3://{bucket_name}/{latest_key}'

    print(f"特徴量をS3に保存中: {s3_path}")

    # タイムスタンプ版
    with smart_open(s3_path, 'wb') as f:
        df.to_parquet(f, index=False)

    # 最新版
    with smart_open(s3_latest_path, 'wb') as f:
        df.to_parquet(f, index=False)

    print("保存完了")
    print(f"  タイムスタンプ版: {s3_path}")
    print(f"  最新版: {s3_latest_path}")
    return s3_path, s3_latest_path

def main():
    """メイン処理"""
    try:
        # データ読み込み
        df = load_data_from_local()

        # 特徴量作成
        df_with_features = _add_timeseries_features(df, window_size_config)

        # 新しい曜日特徴量を確認
        dow_features = [col for col in df_with_features.columns if 'dow' in col.lower()]
        print("\n追加された曜日関連の特徴量:")
        for feat in sorted(dow_features):
            print(f"  - {feat}")

        # S3に保存
        s3_path = save_features_to_s3(df_with_features)

        # ローカルにも保存
        local_path = 'features_enhanced_dow.csv'
        df_with_features.to_csv(local_path, index=False)
        print(f"\nローカルにも保存: {local_path}")

        return df_with_features

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    df = main()