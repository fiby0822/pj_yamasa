#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""学習・予測実行スクリプト（ローカル版）"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import boto3
from io import BytesIO

# 環境変数を読み込み
load_dotenv()

def train_simple_model(df_features, target_col='actual_value'):
    """シンプルなLightGBMモデルの学習"""

    # 特徴量カラムを選択（_fで終わるカラムのみ）
    feature_cols = [col for col in df_features.columns if col.endswith('_f')]

    # 数値型のみを選択
    numeric_cols = []
    for col in feature_cols:
        if col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
            numeric_cols.append(col)

    feature_cols = numeric_cols
    print(f"\n=== 使用する特徴量: {len(feature_cols)}個 ===")
    print(f"特徴量リスト: {feature_cols}")

    # ターゲット変数の処理
    if target_col in df_features.columns:
        df_features[target_col] = pd.to_numeric(df_features[target_col], errors='coerce').fillna(0)
    else:
        print(f"警告: ターゲット変数 '{target_col}' が見つかりません")
        return None, None, None

    # NaNを削除
    df_clean = df_features[feature_cols + [target_col]].dropna()

    if len(df_clean) == 0:
        print("エラー: 有効なデータがありません")
        return None, None, None

    print(f"学習データサイズ: {df_clean.shape}")

    # データを分割
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # LightGBMモデルの学習
    print("\nLightGBMモデル学習中...")

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 100
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )

    # 予測
    y_pred = model.predict(X_test)

    # 評価
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n=== モデル評価結果 ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # 特徴量重要度
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n=== 特徴量重要度 TOP10 ===")
    print(importance.head(10).to_string(index=False))

    return model, importance, {'rmse': rmse, 'mae': mae}

def main():
    """メイン処理"""

    print("="*50)
    print("学習・予測処理開始（ローカル版）")
    print("="*50)

    # S3から特徴量ファイルを読み込む
    bucket_name = "fiby-yamasa-prediction"
    features_key = "features/df_features_yamasa_latest.parquet"

    print(f"\n1. 特徴量データ読込中...")
    print(f"ソース: s3://{bucket_name}/{features_key}")

    # S3クライアント設定
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

    try:
        # S3からデータを読み込み
        obj = s3.get_object(Bucket=bucket_name, Key=features_key)
        df_features = pd.read_parquet(BytesIO(obj['Body'].read()))
        print(f"S3からの読み込み成功")
    except Exception as e:
        print(f"S3からの読み込み失敗: {e}")
        # フォールバックとしてローカルファイルを読み込み
        local_file = "output_data/features/df_features_yamasa_latest.parquet"
        if os.path.exists(local_file):
            print(f"ローカルファイルから読み込み: {local_file}")
            df_features = pd.read_parquet(local_file)
        else:
            print("エラー: データが見つかりません")
            return None
    print(f"読込データサイズ: {df_features.shape}")

    # データの基本情報を表示
    print(f"\n=== データ情報 ===")
    print(f"行数: {len(df_features):,}")
    print(f"カラム数: {len(df_features.columns)}")

    # Material Key毎のデータ数
    if 'material_key' in df_features.columns:
        n_materials = df_features['material_key'].nunique()
        print(f"ユニークなMaterial Key数: {n_materials:,}")

    # 日付範囲
    if 'file_date' in df_features.columns:
        df_features['file_date'] = pd.to_datetime(df_features['file_date'], errors='coerce')
        date_min = df_features['file_date'].min()
        date_max = df_features['file_date'].max()
        print(f"日付範囲: {date_min} 〜 {date_max}")

    # 2. モデル学習
    print(f"\n2. モデル学習開始...")

    model, importance, metrics = train_simple_model(df_features)

    if model is None:
        print("エラー: モデル学習に失敗しました")
        return None

    # 3. 結果をS3とローカルに保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    date_dir = datetime.now().strftime('%Y%m%d')  # YYYYMMDD形式の日付ディレクトリ

    # モデルを保存
    import joblib

    # S3に保存
    print("\nS3に結果を保存中...")

    # モデルをバイト列にシリアライズ
    model_buffer = BytesIO()
    joblib.dump(model, model_buffer)
    model_buffer.seek(0)

    # S3にモデルをアップロード（日付ディレクトリ付き）
    model_key = f"models/{date_dir}/model_{timestamp}.pkl"
    s3.put_object(
        Bucket=bucket_name,
        Key=model_key,
        Body=model_buffer.getvalue()
    )
    print(f"S3モデル保存: s3://{bucket_name}/{model_key}")

    # 最新版としても保存
    model_latest_key = "models/model_latest.pkl"
    s3.put_object(
        Bucket=bucket_name,
        Key=model_latest_key,
        Body=model_buffer.getvalue()
    )
    print(f"S3最新モデル保存: s3://{bucket_name}/{model_latest_key}")

    # 特徴量重要度をS3に保存
    importance_buffer = BytesIO()
    importance.to_parquet(importance_buffer, index=False)
    importance_buffer.seek(0)
    importance_key = f"models/{date_dir}/importance_{timestamp}.parquet"
    s3.put_object(
        Bucket=bucket_name,
        Key=importance_key,
        Body=importance_buffer.getvalue()
    )
    print(f"S3特徴量重要度保存: s3://{bucket_name}/{importance_key}")

    # 最新版としても保存
    importance_latest_key = "models/importance_latest.parquet"
    s3.put_object(
        Bucket=bucket_name,
        Key=importance_latest_key,
        Body=importance_buffer.getvalue()
    )
    print(f"S3最新特徴量重要度保存: s3://{bucket_name}/{importance_latest_key}")

    # メトリクスをS3に保存
    metrics_buffer = BytesIO()
    pd.Series(metrics).to_json(metrics_buffer)
    metrics_buffer.seek(0)
    metrics_key = f"models/{date_dir}/metrics_{timestamp}.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=metrics_key,
        Body=metrics_buffer.getvalue()
    )
    print(f"S3評価指標保存: s3://{bucket_name}/{metrics_key}")

    # 最新版としても保存
    metrics_latest_key = "models/metrics_latest.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=metrics_latest_key,
        Body=metrics_buffer.getvalue()
    )
    print(f"S3最新評価指標保存: s3://{bucket_name}/{metrics_latest_key}")

    # ローカル保存は削除（S3のみに保存）

    print("\n" + "="*50)
    print("処理完了！")
    print("="*50)

    return model, importance, metrics

if __name__ == "__main__":
    main()