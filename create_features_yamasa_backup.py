#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""特徴量作成（confirmed_order_demand_yamasa版）"""

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
    """時系列特徴量を追加（ノートブックの_add_timeseries_featuresを再現）"""

    print("="*50)
    print("時系列特徴量作成開始")
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

        # 新しい日付特徴量
        # 営業日フラグ（平日かつ非祝日・非年末）
        # 土日を非営業日とする
        is_weekend = df['file_date'].dt.dayofweek.isin([5, 6])

        # 年末（12/30, 12/31）を非営業日とする
        is_year_end = ((df['file_date'].dt.month == 12) &
                       (df['file_date'].dt.day.isin([30, 31])))

        # jpholidayを使用して正確な日本の祝日を取得
        def is_japan_holiday(date):
            """jpholidayを使って祝日判定"""
            try:
                # 年始休暇（1/2, 1/3）を追加
                if date.month == 1 and date.day in [2, 3]:
                    return True
                # jpholidayで祝日判定
                return jpholiday.is_holiday(date)
            except:
                return False

        # 各日付に対して祝日判定を適用
        is_holiday = df['file_date'].apply(is_japan_holiday)

        # 営業日フラグ（土日・祝日・年末以外が1）
        df['is_business_day_f'] = (~(is_weekend | is_year_end | is_holiday)).astype(int)

        # 冗長な特徴量は作成しない（is_business_day_fに含まれるため）
        # df['is_weekend_f'] = is_weekend.astype(int)  # 削除
        # df['is_holiday_f'] = is_holiday.astype(int)  # 削除
        # df['is_year_end_f'] = is_year_end.astype(int)  # 削除

        # 金曜日フラグ（ピークの日）
        df['is_friday_f'] = (df['file_date'].dt.dayofweek == 4).astype(int)

        # 曜日×月の交互作用
        df['dow_month_interaction_f'] = df['day_of_week_f'] * df['month_f']

        # 週単位のマイルストーン（7,14,21,28日）フラグ
        df['is_weekly_milestone_f'] = df['day_f'].isin([7, 14, 21, 28]).astype(int)

    # material_key毎の特徴量作成
    if 'material_key' in df.columns:
        print("material_key毎の特徴量作成中...")

        # ソート
        df = df.sort_values(['material_key', 'file_date'])

        # ラグ特徴量
        for lag in window_size_config["material_key"]["lag"]:
            df[f'lag_{lag}_f'] = df.groupby('material_key')['actual_value'].shift(lag)

        # 移動平均（リーク防止のためshift(1)を先に適用）
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

        # 週初からの累積平均 (week-to-date mean) - 簡易版
        if 'day_of_week_f' in df.columns:
            # 週初からの累積平均を計算（より効率的な実装）
            # 各週の開始を検出してグループ化
            df['week_start'] = (df['day_of_week_f'] == 0).astype(int).cumsum()
            df['week_to_date_mean_f'] = df.groupby(['material_key', 'week_start'])['actual_value'].transform(
                lambda x: x.expanding().mean().shift(1).fillna(0)
            )
            df = df.drop('week_start', axis=1)

        # Material Key × 曜日の過去平均 - 効率的な実装
        if 'day_of_week_f' in df.columns:
            # 各Material Key × 曜日の組み合わせで累積平均を計算
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
            print("store_code features done")

        # 品目階層1毎 (category_lvl1)
        if 'category_lvl1' in df.columns:
            print("category_lvl1毎の特徴量作成中...")
            df = df.sort_values(['category_lvl1', 'file_date'])

            for lag in window_size_config["category_lvl1"]["lag"]:
                df[f'category_lvl1_lag_{lag}_f'] = df.groupby('category_lvl1')['actual_value'].shift(lag)

            for cumulative_mean in window_size_config["category_lvl1"]["cumulative_mean"]:
                df[f'category_lvl1_cumulative_mean_{cumulative_mean}_f'] = df.groupby('category_lvl1')['actual_value'].transform(
                    lambda x: x.rolling(window=cumulative_mean, min_periods=1).mean().shift(1)
                )
            print("category_lvl1 features done")

        # 品目階層2毎 (category_lvl2)
        if 'category_lvl2' in df.columns:
            print("category_lvl2毎の特徴量作成中...")
            df = df.sort_values(['category_lvl2', 'file_date'])

            for lag in window_size_config["category_lvl2"]["lag"]:
                df[f'category_lvl2_lag_{lag}_f'] = df.groupby('category_lvl2')['actual_value'].shift(lag)

            for cumulative_mean in window_size_config["category_lvl2"]["cumulative_mean"]:
                df[f'category_lvl2_cumulative_mean_{cumulative_mean}_f'] = df.groupby('category_lvl2')['actual_value'].transform(
                    lambda x: x.rolling(window=cumulative_mean, min_periods=1).mean().shift(1)
                )
            print("category_lvl2 features done")

        # 品目階層3毎 (category_lvl3)
        if 'category_lvl3' in df.columns:
            print("category_lvl3毎の特徴量作成中...")
            df = df.sort_values(['category_lvl3', 'file_date'])

            for lag in window_size_config["category_lvl3"]["lag"]:
                df[f'category_lvl3_lag_{lag}_f'] = df.groupby('category_lvl3')['actual_value'].shift(lag)

            for cumulative_mean in window_size_config["category_lvl3"]["cumulative_mean"]:
                df[f'category_lvl3_cumulative_mean_{cumulative_mean}_f'] = df.groupby('category_lvl3')['actual_value'].transform(
                    lambda x: x.rolling(window=cumulative_mean, min_periods=1).mean().shift(1)
                )
            print("category_lvl3 features done")

        # overall(全体)の特徴量作成
        print("overall特徴量作成中...")
        for lag in window_size_config["overall"]["lag"]:
            df[f'overall_lag_{lag}_f'] = df['actual_value'].shift(lag)

        for rolling_mean in window_size_config["overall"]["rolling_mean"]:
            df[f'overall_rolling_mean_{rolling_mean}_f'] = df['actual_value'].shift(1).rolling(window=rolling_mean).mean()

        for rolling_std in window_size_config["overall"]["rolling_std"]:
            df[f'overall_rolling_std_{rolling_std}_f'] = df['actual_value'].shift(1).rolling(window=rolling_std).std()

        for cumulative_mean in window_size_config["overall"]["cumulative_mean"]:
            df[f'overall_cumulative_mean_{cumulative_mean}_f'] = df['actual_value'].rolling(
                window=cumulative_mean, min_periods=1
            ).mean().shift(1)

        print("overall features done")


    print(f"\n時系列特徴量作成完了！")
    print(f"作成された特徴量数(_f): {len([col for col in df.columns if col.endswith('_f')])}")

    return df

def main():
    """メイン処理"""
    # S3の設定
    bucket_name = "fiby-yamasa-prediction"
    input_file_key = "df_confirmed_order_input_yamasa_fill_zero_df_confirmed_order_input_yamasa_fill_zero.csv"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ローカル保存先
    output_dir = "output_data/features"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/df_features_yamasa_{timestamp}.parquet"
    latest_file = f"{output_dir}/df_features_yamasa_latest.parquet"

    print("="*50)
    print("ヤマサ確定注文用特徴量作成")
    print("="*50)

    # S3クライアント設定
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

    try:
        # S3 URLを構築
        s3_url = f"s3://{bucket_name}/{input_file_key}"

        print(f"データ読込中: {s3_url}")
        print("※10GBのファイルなので時間がかかります...")

        # smart_openでストリーミング読み込み
        chunk_size = 50000
        chunks = []

        with smart_open(s3_url, 'r', encoding='utf-16',
                       transport_params={'client': s3}) as f:
            # ヘッダーを読む
            header = f.readline().strip()
            columns = header.split('\t')
            print(f"カラム数: {len(columns)}")

            # チャンクごとに読み込み（100万行まで）
            lines = []
            line_count = 0
            total_count = 0

            for line in f:
                lines.append(line.strip())
                line_count += 1

                if line_count >= chunk_size:
                    # DataFrameに変換
                    data = [line.split('\t') for line in lines]
                    df_chunk = pd.DataFrame(data, columns=columns)

                    # データ型変換
                    if 'Actual Value' in df_chunk.columns:
                        df_chunk['Actual Value'] = pd.to_numeric(df_chunk['Actual Value'], errors='coerce')

                    chunks.append(df_chunk)
                    total_count += line_count
                    print(f"  処理済み: {total_count:,} 行")

                    # サンプルデータ（100万行）で処理
                    if total_count >= 1000000:
                        print("  サンプルデータ（100万行）で処理します")
                        break

                    lines = []
                    line_count = 0

            # 残りのデータを処理
            if lines and total_count < 1000000:
                data = [line.split('\t') for line in lines]
                df_chunk = pd.DataFrame(data, columns=columns)
                if 'Actual Value' in df_chunk.columns:
                    df_chunk['Actual Value'] = pd.to_numeric(df_chunk['Actual Value'], errors='coerce')
                chunks.append(df_chunk)

        # チャンクを結合
        print(f"\nデータ結合中...")
        df = pd.concat(chunks, ignore_index=True)
        print(f"データサイズ: {df.shape}")

        # データ型の最適化
        print("データ型最適化中...")
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'object':
                # 数値に変換できるカラムは変換
                if col in ['Day Of Week Mon1', 'Week Number', 'レコード数']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # 時系列特徴量作成（ノートブックの_add_timeseries_featuresを使用）
        print("\n時系列特徴量作成開始...")
        df_features = _add_timeseries_features(
            df,
            window_size_config,
            model_type="confirmed_order_demand_yamasa"
        )

        print(f"\n特徴量作成完了: {df_features.shape}")
        print(f"カラム数: {len(df_features.columns)}")

        # usage_type毎のmaterial_key数を表示
        print("\n" + "="*50)
        print("usage_type毎のmaterial_key数:")
        print("="*50)
        if 'usage_type' in df_features.columns and 'material_key' in df_features.columns:
            usage_counts = df_features.groupby('usage_type')['material_key'].nunique()
            total_keys = df_features['material_key'].nunique()
            for usage, count in usage_counts.items():
                print(f"  {usage}: {count:,} material_keys ({count/total_keys*100:.1f}%)")
            print(f"  合計: {total_keys:,} material_keys")
        else:
            if 'usage_type' not in df_features.columns:
                print("  警告: usage_typeカラムが存在しません")
            if 'material_key' not in df_features.columns:
                print("  警告: material_keyカラムが存在しません")

        # _fで終わる特徴量のリストを表示
        feature_cols = [col for col in df_features.columns if col.endswith('_f')]
        print(f"\n作成された特徴量（_f）: {len(feature_cols)}個")
        print("主な特徴量:")
        for i, col in enumerate(feature_cols[:20], 1):
            print(f"  {i:2d}. {col}")
        if len(feature_cols) > 20:
            print(f"  ... 他 {len(feature_cols)-20} 個")

        # S3に保存
        print("\nS3保存中...")
        from io import BytesIO

        # S3クライアントは既に作成済み
        s3 = boto3.client('s3',
                          aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                          aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                          region_name=os.getenv('AWS_DEFAULT_REGION'))

        # Parquetバッファを作成
        parquet_buffer = BytesIO()
        df_features.to_parquet(parquet_buffer, index=False, compression='snappy')
        parquet_buffer.seek(0)

        # 日付ディレクトリを作成 (YYYYMMDD)
        date_dir = datetime.now().strftime('%Y%m%d')

        # S3にアップロード（日付ディレクトリ付き）
        s3_output_key = f"features/{date_dir}/df_features_yamasa_{timestamp}.parquet"
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_output_key,
            Body=parquet_buffer.getvalue()
        )
        print(f"S3保存成功: s3://{bucket_name}/{s3_output_key}")

        # 最新版としても保存（features直下に置く）
        s3_latest_key = "features/df_features_yamasa_latest.parquet"
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_latest_key,
            Body=parquet_buffer.getvalue()
        )
        print(f"S3最新版保存成功: s3://{bucket_name}/{s3_latest_key}")

        # ローカル保存は削除（S3のみに保存）

        # データ情報を表示
        print("\n=== データ情報 ===")
        print(f"行数: {len(df_features):,}")
        print(f"カラム数: {len(df_features.columns)}")
        print(f"メモリ使用量: {df_features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print("\n処理完了！")
        return df_features

    except Exception as e:
        print(f"\nエラー発生: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()