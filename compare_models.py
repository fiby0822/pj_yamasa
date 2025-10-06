#!/usr/bin/env python3
"""
書き換え前後のモデル精度比較スクリプト
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

def load_features_from_s3(bucket_name, file_key):
    """S3から特徴量ファイルを読み込み"""
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_parquet(BytesIO(obj['Body'].read()))
        return df
    except Exception as e:
        print(f"S3読み込みエラー: {e}")
        return None

def load_features_from_local(file_path):
    """ローカルから特徴量ファイルを読み込み"""
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"ローカル読み込みエラー: {e}")
        return None

def prepare_data(df):
    """データ準備"""
    # 特徴量カラムを抽出（_fで終わるカラム）
    feature_cols = [col for col in df.columns if col.endswith('_f')]

    # 欠損値を0で埋める
    df[feature_cols] = df[feature_cols].fillna(0)

    # 無限大を最大値/最小値で置換
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], [df[col][np.isfinite(df[col])].max(),
                                                       df[col][np.isfinite(df[col])].min()])

    # actual_valueがあることを確認
    if 'actual_value' not in df.columns:
        print("警告: actual_valueカラムがありません")
        return None, None, None

    # データを準備
    X = df[feature_cols]
    y = df['actual_value']

    # ゼロ以外のデータでフィルタリング（オプション）
    # non_zero_mask = y > 0
    # X = X[non_zero_mask]
    # y = y[non_zero_mask]

    return X, y, feature_cols

def train_and_evaluate(X, y, model_name="Model"):
    """モデルの学習と評価"""
    print(f"\n{'='*50}")
    print(f"{model_name} の学習と評価")
    print(f"{'='*50}")

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    print(f"訓練データ: {len(X_train):,} 行")
    print(f"テストデータ: {len(X_test):,} 行")
    print(f"特徴量数: {X.shape[1]} 個")

    # モデル学習
    print("\nモデル学習中...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 予測
    print("予測中...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 評価指標計算
    metrics = {
        'model_name': model_name,
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_r2': r2_score(y_test, y_pred_test),
        'n_features': X.shape[1],
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

    # 結果表示
    print(f"\n訓練データの評価:")
    print(f"  RMSE: {metrics['train_rmse']:.2f}")
    print(f"  MAE: {metrics['train_mae']:.2f}")
    print(f"  R2: {metrics['train_r2']:.4f}")

    print(f"\nテストデータの評価:")
    print(f"  RMSE: {metrics['test_rmse']:.2f}")
    print(f"  MAE: {metrics['test_mae']:.2f}")
    print(f"  R2: {metrics['test_r2']:.4f}")

    # 特徴量重要度（上位10個）
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n特徴量重要度（上位10）:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return metrics, model, feature_importance

def main():
    """メイン処理"""
    print("="*70)
    print("モデル精度比較: 書き換え前 vs 書き換え後")
    print("="*70)

    bucket_name = "fiby-yamasa-prediction"
    results = []

    # 1. 書き換え前（オリジナル）のモデル
    print("\n" + "="*70)
    print("1. 書き換え前（オリジナル）のモデル評価")
    print("="*70)

    # ローカルファイルから読み込み試行
    original_paths = [
        "output_data/features/df_features_yamasa_latest.parquet",
        "output_data/features/df_features_yamasa_20241004_155244.parquet",
    ]

    df_original = None
    for path in original_paths:
        if os.path.exists(path):
            print(f"ファイル読込中: {path}")
            df_original = load_features_from_local(path)
            if df_original is not None:
                break

    if df_original is None:
        # S3から読み込み
        s3_key = "output/features/df_features_yamasa_latest.parquet"
        print(f"S3から読込中: s3://{bucket_name}/{s3_key}")
        df_original = load_features_from_s3(bucket_name, s3_key)

    if df_original is not None:
        print(f"データ読込成功: {len(df_original):,} 行")
        X_orig, y_orig, feat_cols_orig = prepare_data(df_original)
        if X_orig is not None:
            metrics_orig, model_orig, importance_orig = train_and_evaluate(
                X_orig, y_orig, "オリジナルモデル（書き換え前）"
            )
            results.append(metrics_orig)
    else:
        print("警告: オリジナルデータが読み込めませんでした")

    # 2. 書き換え後（V2）のモデル
    print("\n" + "="*70)
    print("2. 書き換え後（V2）のモデル評価")
    print("="*70)

    # ローカルファイルから読み込み試行
    v2_paths = [
        "output_data/features/df_features_yamasa_v2_latest.parquet",
    ]

    df_v2 = None
    for path in v2_paths:
        if os.path.exists(path):
            print(f"ファイル読込中: {path}")
            df_v2 = load_features_from_local(path)
            if df_v2 is not None:
                break

    if df_v2 is None:
        # S3から読み込み
        s3_key = "output/features/df_features_yamasa_v2_latest.parquet"
        print(f"S3から読込中: s3://{bucket_name}/{s3_key}")
        df_v2 = load_features_from_s3(bucket_name, s3_key)

    if df_v2 is not None:
        print(f"データ読込成功: {len(df_v2):,} 行")
        X_v2, y_v2, feat_cols_v2 = prepare_data(df_v2)
        if X_v2 is not None:
            metrics_v2, model_v2, importance_v2 = train_and_evaluate(
                X_v2, y_v2, "V2モデル（書き換え後）"
            )
            results.append(metrics_v2)
    else:
        print("警告: V2データが読み込めませんでした")

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
        if len(results) == 2:
            rmse_improvement = (results[0]['test_rmse'] - results[1]['test_rmse']) / results[0]['test_rmse'] * 100
            mae_improvement = (results[0]['test_mae'] - results[1]['test_mae']) / results[0]['test_mae'] * 100
            r2_improvement = (results[1]['test_r2'] - results[0]['test_r2']) / abs(results[0]['test_r2']) * 100 if results[0]['test_r2'] != 0 else 0

            print("\n【改善率】")
            print(f"  RMSE: {rmse_improvement:+.1f}% {'(改善)' if rmse_improvement > 0 else '(悪化)'}")
            print(f"  MAE: {mae_improvement:+.1f}% {'(改善)' if mae_improvement > 0 else '(悪化)'}")
            print(f"  R2: {r2_improvement:+.1f}% {'(改善)' if r2_improvement > 0 else '(悪化)'}")

            print("\n【データ量の比較】")
            print(f"  オリジナル: {results[0]['n_train'] + results[0]['n_test']:,} 行、{results[0]['n_features']} 特徴量")
            print(f"  V2: {results[1]['n_train'] + results[1]['n_test']:,} 行、{results[1]['n_features']} 特徴量")

        # 結果をCSVに保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"model_comparison_{timestamp}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n結果をCSVに保存: {output_file}")

    print("\n" + "="*70)
    print("比較完了")
    print("="*70)

if __name__ == "__main__":
    main()