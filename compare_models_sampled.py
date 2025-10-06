#!/usr/bin/env python3
"""
書き換え前後のモデル精度比較スクリプト（サンプリング版）
メモリ使用量を抑えるため、データをサンプリングして比較
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import boto3
import os
from io import BytesIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# サンプリング設定
SAMPLE_SIZE = 100000  # 10万行に制限
RANDOM_STATE = 42

def create_features_from_zero_filled(df, sample_size=SAMPLE_SIZE):
    """ゼロ補完済みデータから直接特徴量を作成（簡易版）"""
    print(f"データサンプリング中... ({sample_size:,}行)")

    # ランダムサンプリング
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    # カラム名の正規化
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # 日付処理
    if 'file_date' in df.columns:
        df['file_date'] = pd.to_datetime(df['file_date'], errors='coerce')

    # 数値変換
    if 'actual_value' in df.columns:
        df['actual_value'] = pd.to_numeric(df['actual_value'], errors='coerce').fillna(0)

    print("簡易特徴量作成中...")

    # 簡易的な特徴量作成
    features = []

    # 日付特徴量
    if 'file_date' in df.columns:
        df['year_f'] = df['file_date'].dt.year
        df['month_f'] = df['file_date'].dt.month
        df['day_f'] = df['file_date'].dt.day
        df['day_of_week_f'] = df['file_date'].dt.dayofweek
        df['quarter_f'] = df['file_date'].dt.quarter
        features.extend(['year_f', 'month_f', 'day_f', 'day_of_week_f', 'quarter_f'])

    # material_keyごとの統計量（簡易版）
    if 'material_key' in df.columns and 'actual_value' in df.columns:
        df_grouped = df.groupby('material_key')['actual_value']

        # 基本統計量
        df['material_mean_f'] = df_grouped.transform('mean')
        df['material_std_f'] = df_grouped.transform('std').fillna(0)
        df['material_max_f'] = df_grouped.transform('max')
        df['material_min_f'] = df_grouped.transform('min')

        # Lag特徴量（1日、7日）
        df['lag_1_f'] = df_grouped.shift(1).fillna(0)
        df['lag_7_f'] = df_grouped.shift(7).fillna(0)

        # Rolling特徴量（7日、14日）
        df['rolling_mean_7_f'] = df_grouped.transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df['rolling_mean_14_f'] = df_grouped.transform(
            lambda x: x.rolling(14, min_periods=1).mean()
        )

        features.extend(['material_mean_f', 'material_std_f', 'material_max_f', 'material_min_f',
                        'lag_1_f', 'lag_7_f', 'rolling_mean_7_f', 'rolling_mean_14_f'])

    # store_codeごとの統計量（簡易版）
    if 'store_code' in df.columns and 'actual_value' in df.columns:
        df_grouped = df.groupby('store_code')['actual_value']

        df['store_mean_f'] = df_grouped.transform('mean')
        df['store_std_f'] = df_grouped.transform('std').fillna(0)

        features.extend(['store_mean_f', 'store_std_f'])

    # カテゴリ統計量（もしあれば）
    for cat_col in ['category_lvl_1', 'category_lvl_2', 'category_lvl_3']:
        if cat_col in df.columns and 'actual_value' in df.columns:
            df_grouped = df.groupby(cat_col)['actual_value']
            feature_name = f'{cat_col}_mean_f'
            df[feature_name] = df_grouped.transform('mean')
            features.append(feature_name)

    print(f"作成された特徴量: {len(features)}個")

    return df, features

def compare_models():
    """書き換え前後のモデルを比較"""
    print("="*70)
    print("モデル精度比較: 書き換え前 vs 書き換え後（サンプリング版）")
    print("="*70)

    bucket_name = "fiby-yamasa-prediction"
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

    results = []

    # 1. 書き換え前のデータ（元のCSVファイル）を使用した評価
    print("\n" + "="*70)
    print("1. 書き換え前のデータでの評価")
    print("="*70)

    try:
        # 元のデータを読み込み（サンプリング）
        original_key = "data/df_confirmed_order_input_yamasa.parquet"
        print(f"元データ読込中: s3://{bucket_name}/{original_key}")

        df_original = pd.read_parquet(f"s3://{bucket_name}/{original_key}")
        print(f"元データサイズ: {len(df_original):,} 行")

        # サンプリングして特徴量作成
        df_original, features_orig = create_features_from_zero_filled(df_original, SAMPLE_SIZE)

        # 特徴量とターゲットを準備
        feature_cols = [col for col in features_orig if col in df_original.columns]
        X_orig = df_original[feature_cols].fillna(0)
        y_orig = df_original['actual_value']

        # 無限大を置換
        X_orig = X_orig.replace([np.inf, -np.inf], 0)

        # データ分割
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X_orig, y_orig, test_size=0.2, random_state=42, shuffle=False
        )

        print(f"訓練データ: {len(X_train_orig):,} 行")
        print(f"テストデータ: {len(X_test_orig):,} 行")
        print(f"特徴量数: {X_orig.shape[1]} 個")

        # モデル学習
        print("モデル学習中...")
        model_orig = RandomForestRegressor(
            n_estimators=50,  # 高速化のため削減
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model_orig.fit(X_train_orig, y_train_orig)

        # 予測と評価
        y_pred_orig = model_orig.predict(X_test_orig)

        metrics_orig = {
            'model_name': '書き換え前（元データ）',
            'test_rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'test_mae': mean_absolute_error(y_test_orig, y_pred_orig),
            'test_r2': r2_score(y_test_orig, y_pred_orig),
            'n_features': X_orig.shape[1],
            'n_samples': len(X_orig)
        }

        print(f"\nテストデータの評価:")
        print(f"  RMSE: {metrics_orig['test_rmse']:.2f}")
        print(f"  MAE: {metrics_orig['test_mae']:.2f}")
        print(f"  R2: {metrics_orig['test_r2']:.4f}")

        results.append(metrics_orig)

    except Exception as e:
        print(f"エラー: {e}")

    # 2. 書き換え後のデータ（ゼロ補完済み）での評価
    print("\n" + "="*70)
    print("2. 書き換え後のデータでの評価")
    print("="*70)

    try:
        # ゼロ補完済みデータを読み込み
        zero_filled_key = "data/df_confirmed_order_input_yamasa_fill_zero.parquet"
        print(f"ゼロ補完済みデータ読込中: s3://{bucket_name}/{zero_filled_key}")

        df_v2 = pd.read_parquet(f"s3://{bucket_name}/{zero_filled_key}")
        print(f"ゼロ補完済みデータサイズ: {len(df_v2):,} 行")

        # サンプリングして特徴量作成
        df_v2, features_v2 = create_features_from_zero_filled(df_v2, SAMPLE_SIZE)

        # 特徴量とターゲットを準備
        feature_cols = [col for col in features_v2 if col in df_v2.columns]
        X_v2 = df_v2[feature_cols].fillna(0)
        y_v2 = df_v2['actual_value']

        # 無限大を置換
        X_v2 = X_v2.replace([np.inf, -np.inf], 0)

        # データ分割
        X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(
            X_v2, y_v2, test_size=0.2, random_state=42, shuffle=False
        )

        print(f"訓練データ: {len(X_train_v2):,} 行")
        print(f"テストデータ: {len(X_test_v2):,} 行")
        print(f"特徴量数: {X_v2.shape[1]} 個")

        # モデル学習
        print("モデル学習中...")
        model_v2 = RandomForestRegressor(
            n_estimators=50,  # 高速化のため削減
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model_v2.fit(X_train_v2, y_train_v2)

        # 予測と評価
        y_pred_v2 = model_v2.predict(X_test_v2)

        metrics_v2 = {
            'model_name': '書き換え後（ゼロ補完済み）',
            'test_rmse': np.sqrt(mean_squared_error(y_test_v2, y_pred_v2)),
            'test_mae': mean_absolute_error(y_test_v2, y_pred_v2),
            'test_r2': r2_score(y_test_v2, y_pred_v2),
            'n_features': X_v2.shape[1],
            'n_samples': len(X_v2)
        }

        print(f"\nテストデータの評価:")
        print(f"  RMSE: {metrics_v2['test_rmse']:.2f}")
        print(f"  MAE: {metrics_v2['test_mae']:.2f}")
        print(f"  R2: {metrics_v2['test_r2']:.4f}")

        results.append(metrics_v2)

    except Exception as e:
        print(f"エラー: {e}")

    # 3. 結果比較
    if len(results) == 2:
        print("\n" + "="*70)
        print("精度比較結果")
        print("="*70)

        df_results = pd.DataFrame(results)

        print("\n【テストデータでの比較】")
        print(f"{'モデル':<30} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
        print("-" * 60)
        for _, row in df_results.iterrows():
            print(f"{row['model_name']:<30} {row['test_rmse']:>10.2f} {row['test_mae']:>10.2f} {row['test_r2']:>10.4f}")

        # 改善率計算
        rmse_improvement = (results[0]['test_rmse'] - results[1]['test_rmse']) / results[0]['test_rmse'] * 100
        mae_improvement = (results[0]['test_mae'] - results[1]['test_mae']) / results[0]['test_mae'] * 100
        r2_improvement = (results[1]['test_r2'] - results[0]['test_r2']) / abs(results[0]['test_r2']) * 100 if results[0]['test_r2'] != 0 else 0

        print("\n【改善率】")
        print(f"  RMSE: {rmse_improvement:+.1f}% {'(改善)' if rmse_improvement > 0 else '(悪化)'}")
        print(f"  MAE: {mae_improvement:+.1f}% {'(改善)' if mae_improvement > 0 else '(悪化)'}")
        print(f"  R2: {r2_improvement:+.1f}% {'(改善)' if r2_improvement > 0 else '(悪化)'}")

        print("\n【サンプル数と特徴量数】")
        print(f"  書き換え前: {results[0]['n_samples']:,} 行、{results[0]['n_features']} 特徴量")
        print(f"  書き換え後: {results[1]['n_samples']:,} 行、{results[1]['n_features']} 特徴量")

        # 結果をCSVに保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"model_comparison_sampled_{timestamp}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n結果をCSVに保存: {output_file}")

    print("\n" + "="*70)
    print("比較完了")
    print("="*70)

if __name__ == "__main__":
    compare_models()