#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特徴量作成用スクリプト
S3からデータを読み込み、時系列特徴量を追加して保存する
"""

import pandas as pd
import numpy as np
import boto3
from io import StringIO
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

# window_size_config の定義
window_size_config = {
    "material_key": {
        "lag": [1,2,3],
        "rolling_mean": [2,3,4,5,6],
        "rolling_std": [2,3,4,5,6],
        "rate_of_change": [1],
        "cumulative_mean": [2,3,4,5,6, 9,12],
    },
    "variety": {
        "lag": [1,2,3],
        "rolling_mean": [2,3,4,5,6],
        "rolling_std": [2,3,4,5,6],
        "rate_of_change": [1],
        "cumulative_mean": [2,3,4,5,6, 9,12],
    },
    "mill": {
        "lag": [1,2,3],
        "rolling_mean": [2,3,4,5,6],
        "rolling_std": [2,3,4,5,6],
        "rate_of_change": [1],
        "cumulative_mean": [2,3,4,5,6, 9,12],
    },
    "orderer": {
        "lag": [1,2,3],
        "rolling_mean": [2,3,4,5,6],
        "rolling_std": [2,3,4,5,6],
        "rate_of_change": [1],
        "cumulative_mean": [2,3,4,5,6, 9,12],
    },
    "base_code": {
        "cumulative_mean": [2,3,4,5,6, 9,12],
    },
    "customer_code": {
        "cumulative_mean": [2,3,4,5,6, 9,12],
    },
    "primary_consumer_code": {
        "cumulative_mean": [2,3,4,5,6, 9,12],
    },
    "delivery_code": {
        "cumulative_mean": [2,3,4,5,6],
    },
    "place": {
        "cumulative_mean": [2,3,4,5,6],
    },
    "overall": {
        "lag": [1,2,3],
        "rolling_mean": [2,3,4,5,6],
        "rolling_std": [2,3,4,5,6],
        "rate_of_change": [1],
        "cumulative_mean": [2,3,4,5,6],
    },
    # 確定注文予測(ヤマサ)で使用
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
}

def _add_timeseries_features(_df, window_size_config, start_year, end_year, model_type):
    """
    ラグなど時系列関連の特徴量を追加する関数｡
    material_key × file_date毎にactual_value(実績値)を持つデータフレームを想定
    """
    df = _df.copy()
    # データフレームをmaterial_keyとyyyymmでソート
    df.sort_values(by=['material_key', 'file_date'], inplace=True)
    df['file_date'] = pd.to_datetime(_df['file_date'], format='%Y-%m-%d', errors='coerce')
    df['year'] = df['file_date'].dt.year
    df['month'] = df['file_date'].dt.month
    df["year"]  = df["year"].astype("int16")
    df["month"] = df["month"].astype("int8")

    # material_key毎の特徴量作成
    # ラグ特徴量
    for lag in window_size_config["material_key"]["lag"]:
        df[f'lag_{lag}_f'] = df.groupby('material_key')['actual_value'].shift(lag)

    # 移動平均の作成（リーク防止のため、その時点の実績値を除外）
    for rolling_mean in window_size_config["material_key"]["rolling_mean"]:
        df[f'rolling_mean_{rolling_mean}_f'] = df.groupby('material_key')['actual_value'].shift(1).transform(lambda x: x.rolling(window=rolling_mean).mean())

    # 移動標準偏差の作成（リーク防止のため、その時点の実績値を除外）
    for rolling_std in window_size_config["material_key"]["rolling_std"]:
        df[f'rolling_std_{rolling_std}_f'] = df.groupby('material_key')['actual_value'].shift(1).transform(lambda x: x.rolling(window=rolling_std).std())

    # 変動率の作成（リーク防止のため、その時点の実績値を除外）
    for rate_of_change in window_size_config["material_key"]["rate_of_change"]:
        shifted_values = df.groupby('material_key')['actual_value'].shift(1)
        # 変動率の計算前にデータが存在するか確認
        if not shifted_values.empty:
            df[f'rate_of_change_{rate_of_change}_f'] = shifted_values.pct_change()
        else:
            df[f'rate_of_change_{rate_of_change}_f'] = np.nan  # 空の場合はNaNにする

    # 累積平均の作成（リーク防止のため、その時点の実績値を除外）
    for cumulative_mean in window_size_config["material_key"]["cumulative_mean"]:
        df[f'cumulative_mean_{cumulative_mean}_f'] = df.groupby('material_key')['actual_value'].transform(lambda x: x.rolling(window=cumulative_mean, min_periods=1).mean().shift(1))

    print("material_key features done")

    # 確定注文予測(ヤマサ)用の特徴量作成
    if model_type == "confirmed_order_demand_yamasa":

        # 直近に近いほど重く評価する場合（単調減少）
        def weighted_mean_decreasing(y):
            if len(y) == 0:
                return np.nan
            # 単調減少の重み（範囲 -0.8 から 1.2）
            weights = np.linspace(1.2, -0.8, len(y))
            # 中心を1にスケール
            if len(weights) > 0:
                weights /= weights[len(weights) // 2]
            # 重み付き平均を計算
            return np.average(y, weights=weights)

        # 直近に近いほど軽く評価する場合（単調増加）
        def weighted_mean_increasing(y):
            if len(y) == 0:
                return np.nan
            # 単調増加の重み（範囲 -0.8 から 1.2）
            weights = np.linspace(-0.8, 1.2, len(y))
            # 中心を1にスケール
            if len(weights) > 0:
                weights /= weights[len(weights) // 2]
            # 重み付き平均を計算
            return np.average(y, weights=weights)

        # product_key毎の特徴量作成
        if 'product_key' in df.columns:
            for lag in window_size_config["product_key"]["lag"]:
                df[f'product_key_lag_{lag}_f'] = df.groupby('product_key')['actual_value'].shift(lag)

            for cumulative_mean in window_size_config["product_key"]["cumulative_mean"]:
                df[f'product_key_cumulative_mean_{cumulative_mean}_f'] = df.groupby('product_key')['actual_value'].transform(lambda x: x.rolling(window=cumulative_mean, min_periods=1).mean().shift(1))

            # 重み付き累積平均
            for cumulative_mean in window_size_config["product_key"]["cumulative_mean"]:
                df[f'product_key_weighted_cumulative_mean_high_{cumulative_mean}_f'] = df.groupby('product_key')['actual_value'].transform(
                    lambda x: x.rolling(window=cumulative_mean, min_periods=1).apply(weighted_mean_decreasing).shift(1)
                )
                df[f'product_key_weighted_cumulative_mean_low_{cumulative_mean}_f'] = df.groupby('product_key')['actual_value'].transform(
                    lambda x: x.rolling(window=cumulative_mean, min_periods=1).apply(weighted_mean_increasing).shift(1)
                )
            print("product_key features done")

        # store_code毎の特徴量作成
        if 'store_code' in df.columns:
            for lag in window_size_config["store_code"]["lag"]:
                df[f'store_code_lag_{lag}_f'] = df.groupby('store_code')['actual_value'].shift(lag)

            for cumulative_mean in window_size_config["store_code"]["cumulative_mean"]:
                df[f'store_code_cumulative_mean_{cumulative_mean}_f'] = df.groupby('store_code')['actual_value'].transform(lambda x: x.rolling(window=cumulative_mean, min_periods=1).mean().shift(1))

            # 重み付き累積平均
            for cumulative_mean in window_size_config["store_code"]["cumulative_mean"]:
                df[f'store_code_weighted_cumulative_mean_high_{cumulative_mean}_f'] = df.groupby('store_code')['actual_value'].transform(
                    lambda x: x.rolling(window=cumulative_mean, min_periods=1).apply(weighted_mean_decreasing).shift(1)
                )
                df[f'store_code_weighted_cumulative_mean_low_{cumulative_mean}_f'] = df.groupby('store_code')['actual_value'].transform(
                    lambda x: x.rolling(window=cumulative_mean, min_periods=1).apply(weighted_mean_increasing).shift(1)
                )
            print("store_code features done")

        # category_lvl1毎の特徴量作成
        if 'category_lvl1' in df.columns:
            for lag in window_size_config["category_lvl1"]["lag"]:
                df[f'category_lvl1_lag_{lag}_f'] = df.groupby('category_lvl1')['actual_value'].shift(lag)

            for cumulative_mean in window_size_config["category_lvl1"]["cumulative_mean"]:
                df[f'category_lvl1_cumulative_mean_{cumulative_mean}_f'] = df.groupby('category_lvl1')['actual_value'].transform(lambda x: x.rolling(window=cumulative_mean, min_periods=1).mean().shift(1))
            print("category_lvl1 features done")

        # category_lvl2毎の特徴量作成
        if 'category_lvl2' in df.columns:
            for lag in window_size_config["category_lvl2"]["lag"]:
                df[f'category_lvl2_lag_{lag}_f'] = df.groupby('category_lvl2')['actual_value'].shift(lag)

            for cumulative_mean in window_size_config["category_lvl2"]["cumulative_mean"]:
                df[f'category_lvl2_cumulative_mean_{cumulative_mean}_f'] = df.groupby('category_lvl2')['actual_value'].transform(lambda x: x.rolling(window=cumulative_mean, min_periods=1).mean().shift(1))
            print("category_lvl2 features done")

        # category_lvl3毎の特徴量作成
        if 'category_lvl3' in df.columns:
            for lag in window_size_config["category_lvl3"]["lag"]:
                df[f'category_lvl3_lag_{lag}_f'] = df.groupby('category_lvl3')['actual_value'].shift(lag)

            for cumulative_mean in window_size_config["category_lvl3"]["cumulative_mean"]:
                df[f'category_lvl3_cumulative_mean_{cumulative_mean}_f'] = df.groupby('category_lvl3')['actual_value'].transform(lambda x: x.rolling(window=cumulative_mean, min_periods=1).mean().shift(1))
            print("category_lvl3 features done")

        # overall（全体）の特徴量作成
        for lag in window_size_config["overall"]["lag"]:
            df[f'overall_lag_{lag}_f'] = df['actual_value'].shift(lag)

        for rolling_mean in window_size_config["overall"]["rolling_mean"]:
            df[f'overall_rolling_mean_{rolling_mean}_f'] = df['actual_value'].shift(1).rolling(window=rolling_mean).mean()

        for rolling_std in window_size_config["overall"]["rolling_std"]:
            df[f'overall_rolling_std_{rolling_std}_f'] = df['actual_value'].shift(1).rolling(window=rolling_std).std()

        for rate_of_change in window_size_config["overall"]["rate_of_change"]:
            shifted_values = df['actual_value'].shift(1)
            if not shifted_values.empty:
                df[f'overall_rate_of_change_{rate_of_change}_f'] = shifted_values.pct_change()
            else:
                df[f'overall_rate_of_change_{rate_of_change}_f'] = np.nan

        for cumulative_mean in window_size_config["overall"]["cumulative_mean"]:
            df[f'overall_cumulative_mean_{cumulative_mean}_f'] = df['actual_value'].rolling(window=cumulative_mean, min_periods=1).mean().shift(1)

        print("overall features done")

    # 月、曜日の特徴量追加
    df['month_f'] = df['month']
    df['day_of_week_mon1_f'] = df['file_date'].dt.dayofweek  # 月曜=0

    return df

def read_from_s3(bucket_name, file_key):
    """S3からCSVファイルを読み込む"""
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        print(f"データ読込成功: s3://{bucket_name}/{file_key}")
        return df
    except Exception as e:
        print(f"S3からの読み込みエラー: {e}")
        raise

def save_to_s3(df, bucket_name, file_key):
    """DataFrameをS3に保存（Parquet形式）"""
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))
    try:
        # Parquetファイルとして保存
        parquet_buffer = df.to_parquet(index=False)
        s3.put_object(Bucket=bucket_name, Key=file_key, Body=parquet_buffer)
        print(f"データ保存成功: s3://{bucket_name}/{file_key}")
    except Exception as e:
        print(f"S3への保存エラー: {e}")
        raise

def main():
    """メイン処理"""
    # S3の設定
    bucket_name = "fiby-yamasa-prediction"
    input_file_key = "df_confirmed_order_input_yamasa_fill_zero_df_confirmed_order_input_yamasa_fill_zero.csv"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file_key = f"output_data/features/df_features_yamasa_{timestamp}.parquet"
    # 最新の特徴量ファイルとして保存（train_predict.pyで使用）
    latest_file_key = "output_data/features/df_features_yamasa_latest.parquet"

    print("=" * 50)
    print("特徴量作成処理開始")
    print("=" * 50)

    # S3からデータ読込
    print(f"\n1. データ読込中...")
    df_input = read_from_s3(bucket_name, input_file_key)
    print(f"読込データサイズ: {df_input.shape}")
    print(f"カラム: {list(df_input.columns)}")

    # 特徴量作成
    print(f"\n2. 特徴量作成中...")
    df_features = _add_timeseries_features(
        df_input,
        window_size_config,
        start_year=2021,
        end_year=2025,
        model_type="confirmed_order_demand_yamasa"
    )
    print(f"作成後のデータサイズ: {df_features.shape}")
    print(f"追加された特徴量数: {df_features.shape[1] - df_input.shape[1]}")

    # S3に保存
    print(f"\n3. データ保存中...")
    # タイムスタンプ付きファイルとして保存（履歴管理用）
    save_to_s3(df_features, bucket_name, output_file_key)

    # 最新版としても保存（train_predict.pyが参照）
    save_to_s3(df_features, bucket_name, latest_file_key)
    print(f"最新版保存成功: s3://{bucket_name}/{latest_file_key}")

    # ローカル保存はオプションに変更（ディスク容量節約のため）
    save_local = os.getenv('SAVE_LOCAL', 'false').lower() == 'true'
    if save_local:
        local_output_path = f"output/df_features_yamasa_{timestamp}.parquet"
        local_latest_path = "output/df_features_yamasa_latest.parquet"
        os.makedirs("output", exist_ok=True)
        df_features.to_parquet(local_output_path, index=False)
        df_features.to_parquet(local_latest_path, index=False)
        print(f"ローカル保存成功: {local_output_path}")
        print(f"ローカル最新版保存成功: {local_latest_path}")
    else:
        print("ローカル保存はスキップ（SAVE_LOCAL=true で有効化）")

    print("\n" + "=" * 50)
    print("特徴量作成処理完了")
    print("=" * 50)

    # 作成された特徴量のサマリー
    print("\n特徴量サマリー:")
    feature_cols = [col for col in df_features.columns if col.endswith('_f')]
    print(f"特徴量数: {len(feature_cols)}")
    print(f"最初の10個の特徴量: {feature_cols[:10]}")

    # メタデータも保存（特徴量作成時の情報）
    metadata = {
        "created_at": timestamp,
        "input_file": input_file_key,
        "output_file": output_file_key,
        "latest_file": latest_file_key,
        "shape": list(df_features.shape),
        "columns": list(df_features.columns),
        "feature_columns": [col for col in df_features.columns if col.endswith('_f')],
        "n_features": len([col for col in df_features.columns if col.endswith('_f')]),
        "n_records": len(df_features),
        "n_materials": df_features['material_key'].nunique() if 'material_key' in df_features.columns else 0,
        "date_range": {
            "min": str(df_features['file_date'].min()) if 'file_date' in df_features.columns else None,
            "max": str(df_features['file_date'].max()) if 'file_date' in df_features.columns else None
        }
    }

    metadata_key = f"output_data/features/metadata_{timestamp}.json"
    save_json_to_s3(metadata, bucket_name, metadata_key)

    # 最新版のメタデータも保存
    save_json_to_s3(metadata, bucket_name, "output_data/features/metadata_latest.json")
    print(f"メタデータ保存成功: s3://{bucket_name}/{metadata_key}")

    return df_features, output_file_key, latest_file_key

def save_json_to_s3(data, bucket_name, file_key):
    """JSONデータをS3に保存"""
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))
    try:
        import json
        json_str = json.dumps(data, indent=2, default=str, ensure_ascii=False)
        s3.put_object(Bucket=bucket_name, Key=file_key, Body=json_str)
        print(f"JSON保存成功: s3://{bucket_name}/{file_key}")
    except Exception as e:
        print(f"S3への保存エラー: {e}")
        raise

if __name__ == "__main__":
    df_features, output_key, latest_key = main()
    print(f"\n次のステップで使用するファイル:")
    print(f"  - 最新版: {latest_key}")
    print(f"  - タイムスタンプ版: {output_key}")