#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""学習・予測実行スクリプト（usage_type別モデル版）"""

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
import json
import pickle

# 環境変数を読み込み
load_dotenv()

def calculate_prediction_errors(y_true, y_pred, test_meta):
    """予測誤差を計算し、Material Key毎に集計"""

    # 結果をDataFrameにまとめる
    df_results = pd.DataFrame({
        'material_key': test_meta['material_key'],
        'file_date': test_meta['file_date'],
        'usage_type': test_meta.get('usage_type', 'unknown'),
        'actual': y_true,
        'predicted': y_pred,
        'error': y_pred - y_true,
        'abs_error': np.abs(y_pred - y_true),
        'error_rate': np.where(
            y_true != 0,
            (y_pred - y_true) / np.abs(y_true),
            np.where(y_pred != 0, np.inf, 0)
        )
    })

    # Material Key × File Date毎の集計
    df_key_date = df_results.groupby(['material_key', 'file_date', 'usage_type']).agg({
        'actual': 'sum',
        'predicted': 'sum',
        'abs_error': 'sum'
    }).reset_index()

    df_key_date['error_rate'] = np.where(
        df_key_date['actual'] != 0,
        (df_key_date['predicted'] - df_key_date['actual']) / np.abs(df_key_date['actual']),
        np.where(df_key_date['predicted'] != 0, np.inf, 0)
    )

    # Material Key毎の集計（6ヶ月全体）
    df_key_total = df_results.groupby(['material_key', 'usage_type']).agg({
        'actual': 'sum',
        'predicted': 'sum',
        'abs_error': 'sum',
        'error_rate': 'mean'
    }).reset_index()

    # 実績値が正の日数をカウント
    actual_positive_counts = df_results[df_results['actual'] > 0].groupby(['material_key', 'usage_type']).size().reset_index(name='count_actual_positive')
    df_key_total = df_key_total.merge(actual_positive_counts, on=['material_key', 'usage_type'], how='left')
    df_key_total['count_actual_positive_total'] = df_key_total['count_actual_positive'].fillna(0).astype(int)
    df_key_total = df_key_total.drop('count_actual_positive', axis=1)

    # 全体の誤差率を再計算
    df_key_total['error_rate_total'] = np.where(
        df_key_total['actual'] != 0,
        (df_key_total['predicted'] - df_key_total['actual']) / np.abs(df_key_total['actual']),
        np.where(df_key_total['predicted'] != 0, np.inf, 0)
    )

    return df_results, df_key_date, df_key_total

def analyze_error_distribution(df_key_total):
    """誤差率の分布を分析"""

    # 無限大を除外
    df_finite = df_key_total[np.isfinite(df_key_total['error_rate_total'])]

    # 絶対誤差率
    abs_error_rates = np.abs(df_finite['error_rate_total'])

    # 統計量
    error_stats = {
        'within_20_percent': np.sum(abs_error_rates <= 0.2),
        'within_30_percent': np.sum(abs_error_rates <= 0.3),
        'within_50_percent': np.sum(abs_error_rates <= 0.5),
        'total_materials': len(df_finite),
        'error_mean': np.mean(abs_error_rates),
        'error_median': np.median(abs_error_rates),
        'error_std': np.std(abs_error_rates)
    }

    # 割合を計算
    error_stats['within_20_percent_ratio'] = error_stats['within_20_percent'] / error_stats['total_materials']
    error_stats['within_30_percent_ratio'] = error_stats['within_30_percent'] / error_stats['total_materials']
    error_stats['within_50_percent_ratio'] = error_stats['within_50_percent'] / error_stats['total_materials']

    # EVAL形式の新しい評価指標を追加
    df_eval = df_finite.copy()
    df_eval['key_mean_abs_err_div_pred'] = df_eval['abs_error'] / (df_eval['predicted'] + 1e-10)

    error_stats['eval_mean'] = df_eval['key_mean_abs_err_div_pred'].mean()
    error_stats['eval_median'] = df_eval['key_mean_abs_err_div_pred'].median()
    error_stats['eval_total_keys'] = len(df_eval)
    error_stats['eval_within_20'] = np.sum(df_eval['key_mean_abs_err_div_pred'] <= 0.2)
    error_stats['eval_within_30'] = np.sum(df_eval['key_mean_abs_err_div_pred'] <= 0.3)
    error_stats['eval_within_50'] = np.sum(df_eval['key_mean_abs_err_div_pred'] <= 0.5)
    error_stats['eval_within_20_ratio'] = error_stats['eval_within_20'] / error_stats['eval_total_keys']
    error_stats['eval_within_30_ratio'] = error_stats['eval_within_30'] / error_stats['eval_total_keys']
    error_stats['eval_within_50_ratio'] = error_stats['eval_within_50'] / error_stats['eval_total_keys']

    return error_stats

def train_model_for_usage_type(df_train, df_test, usage_type, feature_cols, target_col='actual_value', is_optuna=True):
    """特定のusage_type用のモデルを学習"""

    print(f"\n=== {usage_type}用モデル学習 ===")

    # データ準備
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_test = df_test[feature_cols] if len(df_test) > 0 else pd.DataFrame()
    y_test = df_test[target_col] if len(df_test) > 0 else pd.Series()

    # NaNを0で埋める
    X_train = X_train.fillna(0)
    if len(X_test) > 0:
        X_test = X_test.fillna(0)

    print(f"学習データ: {len(X_train)}件")
    print(f"テストデータ: {len(X_test)}件")

    if len(X_train) == 0:
        print(f"警告: {usage_type}の学習データがありません")
        return None, None, None, None

    # 学習データ内で検証用データを分割
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Optunaによるハイパーパラメータ最適化
    if is_optuna and len(X_tr) > 100:  # データが少ない場合はスキップ
        print(f"Optunaによる最適化中...")

        # S3から最適化済みパラメータを読み込み
        s3 = boto3.client('s3',
                          aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                          aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                          region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))
        bucket_name = "fiby-yamasa-prediction"

        cache_key = f"models/optuna_best_params_{usage_type}.pkl"
        try:
            obj = s3.get_object(Bucket=bucket_name, Key=cache_key)
            best_params = pickle.loads(obj['Body'].read())
            print(f"最適化済みパラメータを使用: {usage_type}")
        except:
            print(f"新規にOptunaで最適化: {usage_type}")

            def objective(trial):
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'verbose': -1,
                    'n_estimators': 100,
                    'random_state': 42
                }

                model_trial = lgb.LGBMRegressor(**params)
                model_trial.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )

                y_pred_val = model_trial.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                return rmse

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20, show_progress_bar=False)

            best_params = {
                'num_leaves': study.best_params['num_leaves'],
                'learning_rate': study.best_params['learning_rate'],
                'feature_fraction': study.best_params['feature_fraction'],
                'bagging_fraction': study.best_params['bagging_fraction'],
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1,
                'n_estimators': 100,
                'random_state': 42
            }

            # S3に保存
            params_buffer = BytesIO()
            params_buffer.write(pickle.dumps(best_params))
            params_buffer.seek(0)
            s3.put_object(
                Bucket=bucket_name,
                Key=cache_key,
                Body=params_buffer.getvalue()
            )
    else:
        # デフォルトパラメータ
        best_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'verbose': -1,
            'n_estimators': 100,
            'random_state': 42
        }

    # モデル学習
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )

    # テストデータで評価
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    else:
        y_pred = np.array([])
        rmse = 0
        mae = 0

    # 特徴量重要度
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_,
        'usage_type': usage_type
    }).sort_values('importance', ascending=False)

    return model, importance, {'rmse': rmse, 'mae': mae}, y_pred

def train_and_predict_with_usage_type(df_features, test_start='2025-01-01', test_end='2025-06-30', target_col='actual_value', is_optuna=True):
    """usage_type別にモデルを学習・予測"""

    # 日付カラムを datetime型に変換
    df_features['file_date'] = pd.to_datetime(df_features['file_date'], errors='coerce')

    # usage_typeカラムがない場合は作成（デフォルト: household）
    if 'usage_type' not in df_features.columns:
        print("警告: usage_typeカラムがないため、全てhouseholdとして処理")
        df_features['usage_type'] = 'household'

    # usage_typeのユニーク値を取得
    usage_types = df_features['usage_type'].unique()
    print(f"\n検出されたusage_type: {usage_types}")

    # テスト期間のデータを分離
    test_mask = (df_features['file_date'] >= test_start) & (df_features['file_date'] <= test_end)
    df_train_all = df_features[~test_mask].copy()
    df_test_all = df_features[test_mask].copy()

    print(f"\n=== データ分割 ===")
    print(f"学習データ期間: {df_train_all['file_date'].min()} 〜 {df_train_all['file_date'].max()}")
    print(f"学習データサイズ: {df_train_all.shape}")
    print(f"テストデータ期間: {df_test_all['file_date'].min()} 〜 {df_test_all['file_date'].max()}")
    print(f"テストデータサイズ: {df_test_all.shape}")

    # 特徴量カラムを選択（_fで終わるカラムのみ）
    feature_cols = [col for col in df_features.columns if col.endswith('_f')]
    print(f"\n使用する特徴量: {len(feature_cols)}個")

    # テストデータから実績値合計が0のMaterial Keyを除外
    df_test_sum = df_test_all.groupby(['material_key', 'usage_type'])[target_col].sum().reset_index()
    valid_keys = df_test_sum[df_test_sum[target_col] > 0][['material_key', 'usage_type']]
    df_test_filtered = df_test_all.merge(valid_keys, on=['material_key', 'usage_type'], how='inner')

    excluded_count = len(df_test_sum) - len(valid_keys)
    print(f"\n=== テストデータのフィルタリング ===")
    print(f"元のテストデータ: {len(df_test_all)}行, {len(df_test_sum)} Material Key×usage_type")
    print(f"フィルタ後: {len(df_test_filtered)}行, {len(valid_keys)} Material Key×usage_type")
    print(f"除外されたMaterial Key×usage_type: {excluded_count}個（実績値合計がゼロ）")

    # usage_type別にモデルを学習
    models = {}
    importances = []
    all_predictions = []
    all_actuals = []
    all_meta = []

    for usage_type in usage_types:
        # usage_typeでフィルタ
        df_train = df_train_all[df_train_all['usage_type'] == usage_type].copy()
        df_test = df_test_filtered[df_test_filtered['usage_type'] == usage_type].copy()

        if len(df_train) == 0:
            print(f"\n{usage_type}: 学習データなしのためスキップ")
            continue

        # モデル学習
        model, importance, metrics, y_pred = train_model_for_usage_type(
            df_train, df_test, usage_type, feature_cols, target_col, is_optuna
        )

        if model is not None:
            models[usage_type] = model
            if importance is not None:
                importances.append(importance)

            if len(y_pred) > 0:
                all_predictions.extend(y_pred)
                all_actuals.extend(df_test[target_col].values)

                # メタデータを保存
                test_meta = pd.DataFrame({
                    'material_key': df_test['material_key'].values,
                    'file_date': df_test['file_date'].values,
                    'usage_type': usage_type
                })
                all_meta.append(test_meta)

    # 結果を結合
    if len(importances) > 0:
        importance_combined = pd.concat(importances, ignore_index=True)
    else:
        importance_combined = pd.DataFrame()

    if len(all_meta) > 0:
        meta_combined = pd.concat(all_meta, ignore_index=True)
    else:
        meta_combined = pd.DataFrame()

    # 全体の評価
    if len(all_predictions) > 0:
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)

        overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        overall_mae = mean_absolute_error(all_actuals, all_predictions)

        print(f"\n=== 全体評価（全usage_type統合） ===")
        print(f"RMSE: {overall_rmse:.4f}")
        print(f"MAE: {overall_mae:.4f}")

        # 予測誤差分析
        df_results, df_key_date, df_key_total = calculate_prediction_errors(
            all_actuals, all_predictions, meta_combined
        )

        # 誤差率分布の分析
        error_stats = analyze_error_distribution(df_key_total)

        # usage_type別の統計も表示
        print(f"\n=== usage_type別の予測精度 ===")
        for ut in df_key_total['usage_type'].unique():
            df_ut = df_key_total[df_key_total['usage_type'] == ut]
            print(f"\n{ut}:")
            print(f"  Material Key数: {len(df_ut)}")
            print(f"  平均誤差率: {np.mean(np.abs(df_ut['error_rate_total']))*100:.2f}%")
            print(f"  中央誤差率: {np.median(np.abs(df_ut['error_rate_total']))*100:.2f}%")
    else:
        overall_rmse = 0
        overall_mae = 0
        error_stats = {}
        df_results = pd.DataFrame()
        df_key_date = pd.DataFrame()
        df_key_total = pd.DataFrame()

    # メトリクスをまとめる
    metrics_combined = {
        'rmse': overall_rmse,
        'mae': overall_mae,
        'error_stats': error_stats
    }

    return models, importance_combined, metrics_combined, (df_results, df_key_date, df_key_total)

def main():
    """メイン処理"""

    print("="*50)
    print("学習・予測処理開始（usage_type別モデル版）")
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

    # S3からデータを読み込み
    obj = s3.get_object(Bucket=bucket_name, Key=features_key)
    df_features = pd.read_parquet(BytesIO(obj['Body'].read()))
    print(f"読込データサイズ: {df_features.shape}")

    # データ情報を表示
    print(f"\n=== データ情報 ===")
    print(f"行数: {len(df_features):,}")
    print(f"カラム数: {len(df_features.columns)}")

    if 'material_key' in df_features.columns:
        n_materials = df_features['material_key'].nunique()
        print(f"ユニークなMaterial Key数: {n_materials:,}")

    if 'usage_type' in df_features.columns:
        print(f"usage_type分布:")
        print(df_features['usage_type'].value_counts())

    # 2. usage_type別モデル学習
    print(f"\n2. usage_type別モデル学習開始...")

    models, importance, metrics, error_analysis = train_and_predict_with_usage_type(
        df_features,
        test_start='2025-01-01',
        test_end='2025-06-30',
        is_optuna=True
    )

    if not models:
        print("エラー: モデル学習に失敗しました")
        return None

    # 3. 結果を保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    date_dir = datetime.now().strftime('%Y%m%d')

    print("\nS3に結果を保存中...")

    # モデルを保存（usage_type別）
    for usage_type, model in models.items():
        model_buffer = BytesIO()
        import joblib
        joblib.dump(model, model_buffer)
        model_buffer.seek(0)

        # タイムスタンプ付き
        model_key = f"models/{date_dir}/model_{usage_type}_{timestamp}.pkl"
        s3.put_object(
            Bucket=bucket_name,
            Key=model_key,
            Body=model_buffer.getvalue()
        )
        print(f"S3モデル保存（{usage_type}）: s3://{bucket_name}/{model_key}")

        # 最新版
        model_latest_key = f"models/model_{usage_type}_latest.pkl"
        s3.put_object(
            Bucket=bucket_name,
            Key=model_latest_key,
            Body=model_buffer.getvalue()
        )

    # 特徴量重要度を保存
    if not importance.empty:
        importance_buffer = BytesIO()
        importance.to_parquet(importance_buffer, index=False)
        importance_buffer.seek(0)
        importance_key = f"models/{date_dir}/importance_usage_type_{timestamp}.parquet"
        s3.put_object(
            Bucket=bucket_name,
            Key=importance_key,
            Body=importance_buffer.getvalue()
        )
        print(f"S3特徴量重要度保存: s3://{bucket_name}/{importance_key}")

        # TOP10を表示
        print(f"\n=== 特徴量重要度 TOP10（usage_type別） ===")
        for ut in importance['usage_type'].unique():
            print(f"\n{ut}:")
            df_ut = importance[importance['usage_type'] == ut].head(10)
            for _, row in df_ut.iterrows():
                print(f"  {row['feature']}: {row['importance']}")

    # エラー分析結果を保存
    if error_analysis:
        df_results, df_key_date, df_key_total = error_analysis

        # Material Key × File Date毎の結果
        key_date_buffer = BytesIO()
        df_key_date.to_parquet(key_date_buffer, index=False)
        key_date_buffer.seek(0)
        key_date_key = f"models/{date_dir}/error_analysis_usage_type_{timestamp}.parquet"
        s3.put_object(
            Bucket=bucket_name,
            Key=key_date_key,
            Body=key_date_buffer.getvalue()
        )
        print(f"S3誤差分析保存: s3://{bucket_name}/{key_date_key}")

        # 最新版
        s3.put_object(
            Bucket=bucket_name,
            Key="models/error_analysis_usage_type_latest.parquet",
            Body=key_date_buffer.getvalue()
        )

    # メトリクスを保存
    metrics_buffer = BytesIO()
    metrics_json = json.dumps(metrics, indent=2, default=str)
    metrics_buffer.write(metrics_json.encode('utf-8'))
    metrics_buffer.seek(0)
    metrics_key = f"models/{date_dir}/metrics_usage_type_{timestamp}.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=metrics_key,
        Body=metrics_buffer.getvalue(),
        ContentType='application/json'
    )
    print(f"S3評価指標保存: s3://{bucket_name}/{metrics_key}")

    print("\n" + "="*50)
    print("処理完了！")
    print("="*50)

    return models, importance, metrics, error_analysis

if __name__ == "__main__":
    main()