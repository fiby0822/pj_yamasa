#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optuna最適化テスト"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
from dotenv import load_dotenv
import boto3
from io import BytesIO
import pickle
import time

load_dotenv()

def test_optuna():
    # S3から特徴量を読み込み
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

    bucket_name = "fiby-yamasa-prediction"
    features_key = "features/df_features_yamasa_latest.parquet"

    print("データ読み込み中...")
    obj = s3.get_object(Bucket=bucket_name, Key=features_key)
    df = pd.read_parquet(BytesIO(obj['Body'].read()))

    # 小さなサンプルで実行
    print("サンプリング中...")
    df = df.sample(n=min(50000, len(df)), random_state=42)

    # 特徴量とターゲットを分離
    feature_cols = [col for col in df.columns if col.endswith('_f')]
    X = df[feature_cols].fillna(0)
    y = df['actual_value'].fillna(0)

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print(f"学習データサイズ: {X_tr.shape}")

    # Optuna最適化
    print("\n=== Optuna最適化開始（試行回数: 10） ===")

    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'verbose': -1,
            'n_estimators': 50,
            'random_state': 42
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(5), lgb.log_evaluation(0)])

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse

    start_time = time.time()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    print(f"\n最適化完了！（{time.time() - start_time:.1f}秒）")
    print(f"最良RMSE: {study.best_value:.4f}")
    print(f"最良パラメータ: {study.best_params}")

    # デフォルトパラメータと比較
    print("\n=== デフォルトパラメータでの学習 ===")
    default_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'verbose': -1,
        'n_estimators': 50
    }

    model_default = lgb.LGBMRegressor(**default_params)
    model_default.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(5), lgb.log_evaluation(0)])

    y_pred_default = model_default.predict(X_test)
    rmse_default = np.sqrt(mean_squared_error(y_test, y_pred_default))

    # 最適化パラメータで学習
    print("\n=== 最適化パラメータでの学習 ===")
    best_params = study.best_params.copy()
    best_params.update({'objective': 'regression', 'metric': 'rmse',
                        'verbose': -1, 'n_estimators': 50})

    model_optuna = lgb.LGBMRegressor(**best_params)
    model_optuna.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(5), lgb.log_evaluation(0)])

    y_pred_optuna = model_optuna.predict(X_test)
    rmse_optuna = np.sqrt(mean_squared_error(y_test, y_pred_optuna))

    print("\n" + "="*50)
    print("📊 Optuna最適化結果比較")
    print("="*50)
    print(f"デフォルトパラメータ RMSE: {rmse_default:.4f}")
    print(f"最適化パラメータ RMSE: {rmse_optuna:.4f}")
    print(f"改善: {rmse_default - rmse_optuna:.4f} ({((rmse_default - rmse_optuna) / rmse_default * 100):.2f}%)")
    print("="*50)

    # S3に最適化パラメータを保存
    best_params_full = study.best_params.copy()
    best_params_full.update({'objective': 'regression', 'metric': 'rmse',
                              'verbose': -1, 'n_estimators': 100, 'random_state': 42})

    params_buffer = BytesIO()
    pickle.dump(best_params_full, params_buffer)
    params_buffer.seek(0)

    s3.put_object(
        Bucket=bucket_name,
        Key="models/optuna_best_params.pkl",
        Body=params_buffer.getvalue()
    )
    print("\n最適化パラメータをS3に保存しました")

    return rmse_default, rmse_optuna

if __name__ == "__main__":
    test_optuna()