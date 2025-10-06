#!/usr/bin/env python3
"""
ヤマサ確定注文データ用の特徴量生成スクリプト（ゼロ補完済みデータ版）
confirmed_order_demand_yamasa用の特徴量を作成
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import boto3
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# 必要に応じてインストール
try:
    import smart_open
except ImportError:
    os.system('pip install smart_open')
    import smart_open

# 使用するカラムの定義
KEY_COLUMNS = ['material_key', 'store_code', 'product_key', 'file_date']
GROUPBY_COLUMNS = ['material_key', 'store_code', 'product_key']
DATE_COLUMN = 'file_date'
VALUE_COLUMN = 'actual_value'

# 対象モデルの設定
MODEL_TYPES = [
    "confirmed_order_demand_yamasa",
]

# ウィンドウサイズ設定（モデルごと）
WINDOW_SIZE_CONFIG = {
    "confirmed_order_demand_yamasa": {
        "lag": [1, 2, 3, 7],
        "rolling_mean": [3, 5, 7, 14, 21, 28],
        "rolling_max": [3, 5, 7],
        "rolling_min": [3, 5, 7],
        "rolling_std": [3, 5, 7, 14],
        "rolling_median": [3, 5, 7],
        "rolling_sum": [3, 5, 7],
        "exponential_mean": [3, 5, 7],
        "rolling_skew": [7],
        "rolling_kurt": [7],
        "diff": [1, 2, 3, 7],
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

        # 営業日フラグ（土日、年末以外）
        df['is_business_day_f'] = (~is_weekend & ~is_year_end).astype(int)

        # 曜日名
        df['day_name_f'] = df['file_date'].dt.day_name()

        # 月のビジネス日数（土日を除く日数）
        df['business_days_in_month_f'] = df['file_date'].apply(
            lambda x: pd.bdate_range(start=x.replace(day=1),
                                    end=(x + pd.offsets.MonthEnd(0))).shape[0]
        )

        # カレンダー上の月の日数
        df['days_in_month_f'] = df['file_date'].dt.days_in_month

        # 前月からの経過日数
        df['days_since_month_start_f'] = df['file_date'].dt.day - 1

        # 月末までの残り日数
        df['days_until_month_end_f'] = df['days_in_month_f'] - df['file_date'].dt.day

        # 前週からの経過日数
        df['days_since_week_start_f'] = df['file_date'].dt.dayofweek

        # 週末までの残り日数
        df['days_until_week_end_f'] = 6 - df['file_date'].dt.dayofweek

        # 半期（上期：1、下期：0）
        df['is_first_half_year_f'] = (df['file_date'].dt.month <= 6).astype(int)

        # 月初週フラグ
        df['is_first_week_of_month_f'] = (df['file_date'].dt.day <= 7).astype(int)

        # 月末週フラグ
        df['is_last_week_of_month_f'] = (
            df['file_date'].dt.day > (df['days_in_month_f'] - 7)
        ).astype(int)

        # 四半期の何番目の月か（1, 2, 3）
        df['month_in_quarter_f'] = ((df['file_date'].dt.month - 1) % 3) + 1

        print(f"  日付特徴量: {19}個")

    # グループごとの時系列特徴量を追加
    features = []
    window_config = window_size_config[model_type]

    # グループを定義（複数の粒度で特徴量を作成）
    group_levels = [
        ['material_key'],
        ['store_code'],
        ['product_key'],
        ['category_lvl_1'],
        ['category_lvl_2'],
        ['category_lvl_3'],
        ['material_key', 'store_code'],
        ['material_key', 'product_key'],
        ['store_code', 'product_key'],
    ]

    print("\n時系列特徴量作成中...")

    for group_cols in group_levels:
        # グループカラムが存在するか確認
        valid_cols = [col for col in group_cols if col in df.columns]
        if not valid_cols:
            continue

        group_name = '_'.join(valid_cols)
        print(f"  グループ: {group_name}")

        # 各グループで特徴量を作成
        for col in valid_cols:
            if col not in df.columns:
                continue

        df_grouped = df.groupby(valid_cols)['actual_value']

        # Lag特徴量
        if 'lag' in window_config:
            for lag in window_config['lag']:
                feature_name = f'{group_name}_lag_{lag}_f'
                df[feature_name] = df_grouped.shift(lag)
                features.append(feature_name)

        # Rolling Mean
        if 'rolling_mean' in window_config:
            for window in window_config['rolling_mean']:
                feature_name = f'{group_name}_rolling_mean_{window}_f'
                df[feature_name] = df_grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                features.append(feature_name)

        # Rolling Max
        if 'rolling_max' in window_config:
            for window in window_config['rolling_max']:
                feature_name = f'{group_name}_rolling_max_{window}_f'
                df[feature_name] = df_grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                features.append(feature_name)

        # Rolling Min
        if 'rolling_min' in window_config:
            for window in window_config['rolling_min']:
                feature_name = f'{group_name}_rolling_min_{window}_f'
                df[feature_name] = df_grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                features.append(feature_name)

        # Rolling Std
        if 'rolling_std' in window_config:
            for window in window_config['rolling_std']:
                feature_name = f'{group_name}_rolling_std_{window}_f'
                df[feature_name] = df_grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                features.append(feature_name)

        # Cumulative Mean（累積平均）
        if 'cumulative_mean' in window_config:
            for window in window_config['cumulative_mean']:
                feature_name = f'{group_name}_cumulative_mean_{window}_f'
                df[feature_name] = df_grouped.transform(
                    lambda x: x.expanding(min_periods=window).mean()
                )
                features.append(feature_name)

    print(f"\n時系列特徴量合計: {len(features)}個")

    # 特徴量の統計情報
    print("\n特徴量統計情報:")
    print(f"  総特徴量数: {len([col for col in df.columns if col.endswith('_f')])}個")
    print(f"  日付特徴量: 19個")
    print(f"  時系列特徴量: {len(features)}個")

    return df

def main():
    """メイン処理"""
    # S3の設定
    bucket_name = "fiby-yamasa-prediction"
    # ゼロ補完済みのparquetファイルを読み込むように変更
    input_file_key = "data/df_confirmed_order_input_yamasa_fill_zero.parquet"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # ローカル保存先
    output_dir = "output_data/features"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/df_features_yamasa_v2_{timestamp}.parquet"
    latest_file = f"{output_dir}/df_features_yamasa_v2_latest.parquet"

    print("="*50)
    print("ヤマサ確定注文用特徴量作成（V2: ゼロ補完済みデータ版）")
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
        print("※Parquetファイルを読み込んでいます...")

        # Parquetファイルを直接読み込み
        df = pd.read_parquet(s3_url)
        print(f"読込完了: {len(df):,} 行")

        # データ型を適切に変換
        print("\nデータ型変換中...")

        # 数値カラムを変換
        numeric_columns = ['actual_value', 'forecast_value', 'deviation',
                          'deviation_percentage', 'accuracy_percentage']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 日付カラムを変換
        date_columns = ['file_date', 'forecast_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        print("データ型変換完了")

        # 基本情報表示
        print("\n=== データ基本情報 ===")
        print(f"行数: {len(df):,}")
        print(f"列数: {len(df.columns)}")
        print(f"\nカラム: {list(df.columns)}")

        # actual_valueの統計
        if 'actual_value' in df.columns:
            print(f"\nactual_valueの統計:")
            print(f"  平均: {df['actual_value'].mean():.2f}")
            print(f"  標準偏差: {df['actual_value'].std():.2f}")
            print(f"  最小: {df['actual_value'].min():.2f}")
            print(f"  最大: {df['actual_value'].max():.2f}")
            print(f"  ゼロの数: {(df['actual_value'] == 0).sum():,} ({(df['actual_value'] == 0).mean()*100:.1f}%)")

        # file_dateの範囲
        if 'file_date' in df.columns:
            print(f"\nfile_dateの範囲:")
            print(f"  開始: {df['file_date'].min()}")
            print(f"  終了: {df['file_date'].max()}")

        # 特徴量作成
        print("\n" + "="*50)
        print("特徴量作成開始")
        print("="*50)

        # 時系列特徴量を追加
        df_features = _add_timeseries_features(
            df.copy(),
            window_size_config=WINDOW_SIZE_CONFIG,
            model_type="confirmed_order_demand_yamasa"
        )

        # 特徴量のみを抽出（_fで終わるカラム）
        feature_cols = [col for col in df_features.columns if col.endswith('_f')]
        print(f"\n作成された特徴量数: {len(feature_cols)}")

        # 必要なカラムを保持
        keep_cols = KEY_COLUMNS + ['actual_value'] + feature_cols
        keep_cols = [col for col in keep_cols if col in df_features.columns]
        df_final = df_features[keep_cols]

        # 保存
        print("\n結果を保存中...")
        df_final.to_parquet(output_file)
        df_final.to_parquet(latest_file)  # latest版も保存
        print(f"✓ 保存完了: {output_file}")
        print(f"✓ 最新版: {latest_file}")

        # S3にもアップロード
        print("\nS3にアップロード中...")
        buffer = BytesIO()
        df_final.to_parquet(buffer)
        buffer.seek(0)

        s3_output_key = f"output/features/df_features_yamasa_v2_{timestamp}.parquet"
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_output_key,
            Body=buffer.getvalue()
        )
        print(f"✓ S3保存完了: s3://{bucket_name}/{s3_output_key}")

        # 最新版もS3に保存
        s3_latest_key = f"output/features/df_features_yamasa_v2_latest.parquet"
        buffer.seek(0)
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_latest_key,
            Body=buffer.getvalue()
        )
        print(f"✓ S3最新版: s3://{bucket_name}/{s3_latest_key}")

        print("\n" + "="*50)
        print("特徴量作成完了")
        print("="*50)

        # 作成結果サマリ
        print("\n=== 作成結果サマリ ===")
        print(f"入力データ: {len(df):,} 行")
        print(f"出力データ: {len(df_final):,} 行")
        print(f"特徴量数: {len(feature_cols)} 個")
        print(f"出力カラム数: {len(df_final.columns)} 個")

        # 特徴量のカテゴリ別集計
        print("\n特徴量カテゴリ別:")
        categories = {}
        for col in feature_cols:
            if 'lag_' in col:
                cat = 'lag'
            elif 'rolling_mean_' in col:
                cat = 'rolling_mean'
            elif 'rolling_max_' in col:
                cat = 'rolling_max'
            elif 'rolling_min_' in col:
                cat = 'rolling_min'
            elif 'rolling_std_' in col:
                cat = 'rolling_std'
            elif 'cumulative_mean_' in col:
                cat = 'cumulative_mean'
            elif any(x in col for x in ['year_f', 'month_f', 'day_f', 'week', 'quarter', 'business']):
                cat = 'date'
            else:
                cat = 'other'

            categories[cat] = categories.get(cat, 0) + 1

        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} 個")

        return df_final

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    df = main()