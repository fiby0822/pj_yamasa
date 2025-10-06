#!/usr/bin/env python3
"""
パターン1用の学習・予測スクリプト
12特徴量を使用した予測モデル（Optuna無し）
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import boto3
import os
from datetime import datetime
from io import BytesIO
import joblib
import warnings
warnings.filterwarnings('ignore')

# 固定パラメータ
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

def load_features():
    """特徴量データの読み込み"""
    # 最新の特徴量ファイルを読み込み
    local_file = "output_data/features/df_features_pattern1_latest.parquet"

    if os.path.exists(local_file):
        print(f"ローカルファイルから読み込み: {local_file}")
        df = pd.read_parquet(local_file)
    else:
        # S3から読み込み
        bucket_name = "fiby-yamasa-prediction"
        s3_key = "output/features/df_features_pattern1_latest.parquet"
        print(f"S3から読み込み: s3://{bucket_name}/{s3_key}")
        df = pd.read_parquet(f"s3://{bucket_name}/{s3_key}")

    return df

def prepare_data(df):
    """データの準備"""
    # 特徴量カラムの特定（_fで終わるカラム）
    feature_cols = [col for col in df.columns if col.endswith('_f')]
    print(f"使用する特徴量: {len(feature_cols)}個")

    # データの準備
    X = df[feature_cols].fillna(0)
    y = df['actual_value'].fillna(0)

    # 無限大の処理
    X = X.replace([np.inf, -np.inf], 0)

    return X, y, feature_cols

def train_model(X_train, y_train, X_test, y_test, save_model=True):
    """モデルの学習"""
    print("\nモデル学習開始...")
    print("モデルパラメータ:")
    for key, value in MODEL_PARAMS.items():
        print(f"  {key}: {value}")

    # モデル学習
    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    # 予測
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 評価メトリクス計算
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    print("\n=== 学習結果 ===")
    print("訓練データ:")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  MAE: {train_mae:.2f}")
    print(f"  R²: {train_r2:.4f}")

    print("\nテストデータ:")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE: {test_mae:.2f}")
    print(f"  R²: {test_r2:.4f}")

    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n特徴量重要度（上位5）:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # モデル保存
    if save_model:
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_file = f"models/model_pattern1_{timestamp}.pkl"
        joblib.dump(model, model_file)
        print(f"\nモデル保存: {model_file}")

        # 最新版も保存
        latest_model = "models/model_pattern1_latest.pkl"
        joblib.dump(model, latest_model)
        print(f"最新版: {latest_model}")

    metrics = {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }

    return model, metrics, y_pred_test, feature_importance

def create_predictions(df, model, feature_cols, y_test, y_pred):
    """予測結果の作成"""
    # テストデータのインデックスを取得
    test_size = len(y_test)
    test_indices = df.index[-test_size:]

    # 予測結果データフレームの作成
    df_pred = df.loc[test_indices, ['material_key', 'store_code', 'product_key', 'file_date']].copy()
    df_pred['actual_value'] = y_test.values
    df_pred['predicted_value'] = y_pred
    df_pred['error'] = df_pred['actual_value'] - df_pred['predicted_value']
    df_pred['abs_error'] = np.abs(df_pred['error'])

    # 誤差率の計算
    with np.errstate(divide='ignore', invalid='ignore'):
        df_pred['error_rate'] = np.abs(df_pred['error']) / np.where(
            df_pred['actual_value'] != 0,
            np.abs(df_pred['actual_value']),
            1
        ) * 100

    # 誤差率の集計
    print("\n=== 予測精度分析 ===")
    print(f"総予測数: {len(df_pred):,}")
    print(f"平均絶対誤差: {df_pred['abs_error'].mean():.2f}")
    print(f"平均誤差率: {df_pred['error_rate'].mean():.2f}%")
    print(f"中央誤差率: {df_pred['error_rate'].median():.2f}%")

    # 誤差率別の集計
    error_thresholds = [20, 30, 50]
    for threshold in error_thresholds:
        count = (df_pred['error_rate'] <= threshold).sum()
        percentage = count / len(df_pred) * 100
        print(f"{threshold}%以内: {count:,}個 ({percentage:.1f}%)")

    return df_pred

def main():
    """メイン処理"""
    print("="*70)
    print("パターン1 モデル学習・予測")
    print("="*70)

    # 特徴量データ読み込み
    print("\n特徴量データ読み込み中...")
    df = load_features()
    print(f"読込完了: {len(df):,} 行 × {len(df.columns)} 列")

    # データ準備
    X, y, feature_cols = prepare_data(df)

    # データ分割（時系列なのでシャッフル無し）
    print("\nデータ分割中...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    print(f"  訓練データ: {len(X_train):,} 行")
    print(f"  テストデータ: {len(X_test):,} 行")

    # モデル学習
    model, metrics, y_pred_test, feature_importance = train_model(
        X_train, y_train, X_test, y_test
    )

    # 予測結果の作成
    df_pred = create_predictions(df, model, feature_cols, y_test, y_pred_test)

    # 結果保存
    print("\n" + "="*70)
    print("結果保存")
    print("="*70)

    os.makedirs("output_data/predictions", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 予測結果を保存
    pred_file = f"output_data/predictions/predictions_pattern1_{timestamp}.parquet"
    df_pred.to_parquet(pred_file, index=False)
    print(f"✓ 予測結果: {pred_file}")

    # CSV版も保存
    csv_file = f"output_data/predictions/predictions_pattern1_{timestamp}.csv"
    df_pred.to_csv(csv_file, index=False)
    print(f"✓ CSV版: {csv_file}")

    # メトリクスを保存
    metrics_df = pd.DataFrame([metrics])
    metrics_file = f"output_data/predictions/metrics_pattern1_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✓ メトリクス: {metrics_file}")

    # 特徴量重要度を保存
    importance_file = f"output_data/predictions/importance_pattern1_{timestamp}.csv"
    feature_importance.to_csv(importance_file, index=False)
    print(f"✓ 特徴量重要度: {importance_file}")

    # S3にもアップロード
    print("\nS3にアップロード中...")
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

    bucket_name = "fiby-yamasa-prediction"

    # 予測結果
    buffer = BytesIO()
    df_pred.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3_pred_key = f"output/predictions/predictions_pattern1_{timestamp}.parquet"
    s3.put_object(Bucket=bucket_name, Key=s3_pred_key, Body=buffer.getvalue())
    print(f"✓ S3予測結果: s3://{bucket_name}/{s3_pred_key}")

    print("\n" + "="*70)
    print("学習・予測完了")
    print("="*70)

    return model, df_pred, metrics

if __name__ == "__main__":
    model, df_pred, metrics = main()