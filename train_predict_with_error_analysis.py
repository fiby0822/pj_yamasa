#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""学習・予測実行スクリプト（予測誤差分析機能付き）"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from datetime import datetime
from dotenv import load_dotenv
import boto3
from io import BytesIO
import pickle

# 環境変数を読み込み
load_dotenv()

def calculate_prediction_errors(y_true, y_pred, test_data, material_key_col='material_key', file_date_col='file_date'):
    """予測誤差を計算

    Args:
        y_true: 実績値
        y_pred: 予測値
        test_data: テストデータ（Material KeyとFile Dateを含む）
        material_key_col: Material Keyのカラム名
        file_date_col: File Dateのカラム名

    Returns:
        誤差分析結果のDataFrame
    """

    # 結果をDataFrameにまとめる
    df_results = pd.DataFrame({
        material_key_col: test_data[material_key_col].values,
        file_date_col: test_data[file_date_col].values,
        'actual': y_true.values,
        'predicted': y_pred,
    })

    # 誤差率を計算（実績値が0でない場合のみ）
    df_results['error_rate'] = np.where(
        df_results['actual'] != 0,
        (df_results['predicted'] - df_results['actual']) / df_results['actual'],
        np.nan
    )

    # 絶対誤差率も計算
    df_results['abs_error_rate'] = np.abs(df_results['error_rate'])

    # Material Key × File Date毎の集計
    df_key_date = df_results.groupby([material_key_col, file_date_col]).agg({
        'actual': 'sum',
        'predicted': 'sum',
        'error_rate': lambda x: x[~x.isna()].mean() if (~x.isna()).any() else np.nan,
        'abs_error_rate': lambda x: x[~x.isna()].mean() if (~x.isna()).any() else np.nan
    }).reset_index()

    # Material Key × File Date毎の誤差率を再計算（合計値ベース）
    df_key_date['error_rate_sum'] = np.where(
        df_key_date['actual'] != 0,
        (df_key_date['predicted'] - df_key_date['actual']) / df_key_date['actual'],
        np.nan
    )

    # 実績値>0のカウント（Material Key × File Date毎）
    df_count = df_results[df_results['actual'] > 0].groupby([material_key_col, file_date_col]).size().reset_index(name='count_actual_positive')
    df_key_date = df_key_date.merge(df_count, on=[material_key_col, file_date_col], how='left')
    df_key_date['count_actual_positive'] = df_key_date['count_actual_positive'].fillna(0).astype(int)

    # Material Key毎の集計（6ヶ月全体）
    df_key_total = df_results.groupby(material_key_col).agg({
        'actual': 'sum',
        'predicted': 'sum',
        'error_rate': lambda x: x[~x.isna()].mean() if (~x.isna()).any() else np.nan,
        'abs_error_rate': lambda x: x[~x.isna()].mean() if (~x.isna()).any() else np.nan
    }).reset_index()

    # Material Key毎の誤差率を再計算（合計値ベース）
    df_key_total['error_rate_total'] = np.where(
        df_key_total['actual'] != 0,
        (df_key_total['predicted'] - df_key_total['actual']) / df_key_total['actual'],
        np.nan
    )

    # 実績値>0のカウント（Material Key毎、6ヶ月全体）
    df_count_total = df_results[df_results['actual'] > 0].groupby(material_key_col).size().reset_index(name='count_actual_positive_total')
    df_key_total = df_key_total.merge(df_count_total, on=material_key_col, how='left')
    df_key_total['count_actual_positive_total'] = df_key_total['count_actual_positive_total'].fillna(0).astype(int)

    # key_mean_abs_err_div_pred_overall の計算
    # 各Material Keyごとに: mean(|actual - predicted|) / mean(predicted)
    df_key_metrics = df_results.groupby(material_key_col).agg({
        'actual': 'mean',
        'predicted': 'mean'
    }).reset_index()
    df_key_metrics.columns = [material_key_col, 'actual_mean', 'predicted_mean']

    # 絶対誤差の平均を計算
    df_abs_error = df_results.copy()
    df_abs_error['abs_error'] = np.abs(df_abs_error['actual'] - df_abs_error['predicted'])
    df_key_abs_error = df_abs_error.groupby(material_key_col)['abs_error'].mean().reset_index()
    df_key_abs_error.columns = [material_key_col, 'mean_abs_error']

    # マージして計算
    df_key_metrics = df_key_metrics.merge(df_key_abs_error, on=material_key_col)
    df_key_metrics['key_mean_abs_err_div_pred_overall'] = np.where(
        df_key_metrics['predicted_mean'] != 0,
        df_key_metrics['mean_abs_error'] / df_key_metrics['predicted_mean'],
        np.nan
    )

    # df_key_totalにマージ
    df_key_total = df_key_total.merge(
        df_key_metrics[[material_key_col, 'key_mean_abs_err_div_pred_overall']],
        on=material_key_col,
        how='left'
    )

    return df_results, df_key_date, df_key_total

def analyze_error_distribution(df_key_total):
    """誤差率の分布を分析

    Args:
        df_key_total: Material Key毎の集計結果

    Returns:
        誤差率別の統計
    """

    # 絶対誤差率で判定（従来の指標）
    df_valid = df_key_total[~df_key_total['error_rate_total'].isna()].copy()
    df_valid['abs_error_rate_total'] = np.abs(df_valid['error_rate_total'])

    total_materials = len(df_valid)

    # 誤差率別のカウント（従来の指標）
    within_20 = (df_valid['abs_error_rate_total'] <= 0.2).sum()
    within_30 = (df_valid['abs_error_rate_total'] <= 0.3).sum()
    within_50 = (df_valid['abs_error_rate_total'] <= 0.5).sum()

    # 誤差率の統計値
    error_mean = df_valid['abs_error_rate_total'].mean()
    error_median = df_valid['abs_error_rate_total'].median()
    error_std = df_valid['abs_error_rate_total'].std()

    # 新しい評価指標: key_mean_abs_err_div_pred_overall
    df_key_eval = df_key_total[~df_key_total['key_mean_abs_err_div_pred_overall'].isna()].copy()
    total_keys_eval = len(df_key_eval)

    # key_mean_abs_err_div_pred_overall ベースの評価
    eval_mean = df_key_eval['key_mean_abs_err_div_pred_overall'].mean()
    eval_median = df_key_eval['key_mean_abs_err_div_pred_overall'].median()
    eval_within_20 = (df_key_eval['key_mean_abs_err_div_pred_overall'] <= 0.2).sum()
    eval_within_30 = (df_key_eval['key_mean_abs_err_div_pred_overall'] <= 0.3).sum()
    eval_within_50 = (df_key_eval['key_mean_abs_err_div_pred_overall'] <= 0.5).sum()

    stats = {
        'total_materials': total_materials,
        'within_20_percent': within_20,
        'within_30_percent': within_30,
        'within_50_percent': within_50,
        'within_20_percent_ratio': within_20 / total_materials if total_materials > 0 else 0,
        'within_30_percent_ratio': within_30 / total_materials if total_materials > 0 else 0,
        'within_50_percent_ratio': within_50 / total_materials if total_materials > 0 else 0,
        'error_mean': error_mean,
        'error_median': error_median,
        'error_std': error_std,
        # 新しい評価指標
        'eval_total_keys': total_keys_eval,
        'eval_mean': eval_mean,
        'eval_median': eval_median,
        'eval_within_20': eval_within_20,
        'eval_within_30': eval_within_30,
        'eval_within_50': eval_within_50,
        'eval_within_20_ratio': eval_within_20 / total_keys_eval if total_keys_eval > 0 else 0,
        'eval_within_30_ratio': eval_within_30 / total_keys_eval if total_keys_eval > 0 else 0,
        'eval_within_50_ratio': eval_within_50 / total_keys_eval if total_keys_eval > 0 else 0
    }

    return stats

def train_and_predict_with_test_period(df_features, test_start='2025-01-01', test_end='2025-06-30', target_col='actual_value', is_optuna=True):
    """指定期間をテストデータとしてモデル学習と予測を実行

    Args:
        df_features: 特徴量データフレーム
        test_start: テスト期間開始日
        test_end: テスト期間終了日
        target_col: ターゲット変数のカラム名
        is_optuna: Optunaによるハイパーパラメータ最適化を行うかどうか（デフォルト: True）
    """

    # 日付カラムを datetime型に変換
    df_features['file_date'] = pd.to_datetime(df_features['file_date'], errors='coerce')

    # テスト期間のデータを分離
    test_mask = (df_features['file_date'] >= test_start) & (df_features['file_date'] <= test_end)
    df_train = df_features[~test_mask].copy()
    df_test = df_features[test_mask].copy()

    print(f"\n=== データ分割 ===")
    print(f"学習データ期間: {df_train['file_date'].min()} 〜 {df_train['file_date'].max()}")
    print(f"学習データサイズ: {df_train.shape}")
    print(f"テストデータ期間: {df_test['file_date'].min()} 〜 {df_test['file_date'].max()}")
    print(f"テストデータサイズ: {df_test.shape}")

    # 特徴量カラムを選択（_fで終わるカラムのみ）
    feature_cols = [col for col in df_features.columns if col.endswith('_f')]

    # 数値型のみを選択
    numeric_cols = []
    for col in feature_cols:
        if col in df_features.columns:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
            numeric_cols.append(col)

    feature_cols = numeric_cols
    print(f"\n=== 使用する特徴量: {len(feature_cols)}個 ===")
    print(f"特徴量リスト: {feature_cols}")

    # ターゲット変数の処理
    if target_col in df_train.columns:
        df_train[target_col] = pd.to_numeric(df_train[target_col], errors='coerce').fillna(0)
        df_test[target_col] = pd.to_numeric(df_test[target_col], errors='coerce').fillna(0)
    else:
        print(f"警告: ターゲット変数 '{target_col}' が見つかりません")
        return None, None, None, None

    # NaNを削除
    df_train_clean = df_train[feature_cols + [target_col]].dropna()
    df_test_clean = df_test[feature_cols + [target_col, 'material_key', 'file_date']].dropna()

    # テストデータから実績値の合計がゼロのmaterial_keyを除外
    print("\n=== テストデータのフィルタリング ===")
    material_key_actual_sum = df_test_clean.groupby('material_key')[target_col].sum()
    material_keys_with_nonzero = material_key_actual_sum[material_key_actual_sum > 0].index

    original_test_size = len(df_test_clean)
    original_material_keys = df_test_clean['material_key'].nunique()

    df_test_clean = df_test_clean[df_test_clean['material_key'].isin(material_keys_with_nonzero)]

    filtered_test_size = len(df_test_clean)
    filtered_material_keys = df_test_clean['material_key'].nunique()

    print(f"元のテストデータ: {original_test_size}行, {original_material_keys} Material Keys")
    print(f"フィルタ後: {filtered_test_size}行, {filtered_material_keys} Material Keys")
    print(f"除外されたMaterial Keys: {original_material_keys - filtered_material_keys}個（実績値合計がゼロ）")

    if len(df_train_clean) == 0 or len(df_test_clean) == 0:
        print("エラー: 有効なデータがありません")
        return None, None, None, None

    # データを準備
    X_train = df_train_clean[feature_cols]
    y_train = df_train_clean[target_col]
    X_test = df_test_clean[feature_cols]
    y_test = df_test_clean[target_col]

    # テストデータのメタ情報を保持
    test_meta = df_test_clean[['material_key', 'file_date']]

    # 検証用データを学習データから分割
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Optunaによるハイパーパラメータ最適化
    if is_optuna:
        print("\n=== Optunaによるハイパーパラメータ最適化中... ===")

        # 最適化済みパラメータのキャッシュファイル
        cache_file = "optuna_best_params_cache.pkl"
        s3 = boto3.client('s3',
                          aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                          aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                          region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))
        bucket_name = "fiby-yamasa-prediction"

        # S3から最適化済みパラメータを読み込み
        try:
            obj = s3.get_object(Bucket=bucket_name, Key="models/optuna_best_params.pkl")
            best_params = pickle.loads(obj['Body'].read())
            print("最適化済みパラメータをS3から読み込みました")
            print(f"使用パラメータ: {best_params}")
        except:
            print("新規にOptunaで最適化を実行します...")

            def objective(trial):
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                    'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
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

            # Optunaによる最適化実行
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20, show_progress_bar=True)  # 試行回数を20に削減

            best_params = study.best_params
            best_params['objective'] = 'regression'
            best_params['metric'] = 'rmse'
            best_params['verbose'] = -1
            best_params['n_estimators'] = 100
            best_params['random_state'] = 42

            print(f"\n最適化完了！")
            print(f"最良パラメータ: {best_params}")
            print(f"最良RMSE: {study.best_value:.4f}")

            # S3に最適化済みパラメータを保存
            params_buffer = BytesIO()
            pickle.dump(best_params, params_buffer)
            params_buffer.seek(0)
            s3.put_object(
                Bucket=bucket_name,
                Key="models/optuna_best_params.pkl",
                Body=params_buffer.getvalue()
            )
            print("最適化済みパラメータをS3に保存しました")

        params = best_params
    else:
        print("\nLightGBMモデル学習中（Optunaなし）...")
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

    # モデル学習
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )

    # テストデータで予測
    y_pred = model.predict(X_test)

    # 基本的な評価指標
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n=== モデル評価結果（テストデータ: {test_start} 〜 {test_end}）===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # 特徴量重要度
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n=== 特徴量重要度 TOP10 ===")
    print(importance.head(10).to_string(index=False))

    # 予測誤差分析
    print(f"\n=== 予測誤差分析 ===")
    df_results, df_key_date, df_key_total = calculate_prediction_errors(
        y_test, y_pred, test_meta
    )

    # 誤差率分布の分析
    error_stats = analyze_error_distribution(df_key_total)

    # 全Material Key数を取得（実績値が0のものは既にテストデータから除外されている）
    total_all_materials = len(df_key_total)

    print(f"\n=== Material Key毎の予測精度分布 ===")
    print(f"分析対象Material Key数: {total_all_materials}個")
    print(f"（注: 実績値合計が0のMaterial Keyは事前に除外済み）")
    print(f"\n予測精度分布:")
    print(f"予測誤差20%以内: {error_stats['within_20_percent']}個 ({error_stats['within_20_percent_ratio']:.1%})")
    print(f"予測誤差30%以内: {error_stats['within_30_percent']}個 ({error_stats['within_30_percent_ratio']:.1%})")
    print(f"予測誤差50%以内: {error_stats['within_50_percent']}個 ({error_stats['within_50_percent_ratio']:.1%})")
    print(f"\n予測誤差率の統計:")
    print(f"平均誤差率: {error_stats['error_mean']*100:.2f}%")
    print(f"中央誤差率: {error_stats['error_median']*100:.2f}%")
    print(f"標準偏差: {error_stats['error_std']*100:.2f}%")

    # 改善結果のサマリー表示（前回との比較）
    print(f"\n=== 📊 予測精度改善サマリー（新特徴量追加による効果） ===")
    print(f"【改善前 → 改善後】")
    print(f"予測誤差平均: 965.96% → {error_stats['error_mean']*100:.2f}%")
    print(f"予測誤差中央値: 224.47% → {error_stats['error_median']*100:.2f}%")
    print(f"20%以内: 89個 (17.8%) → {error_stats['within_20_percent']}個 ({error_stats['within_20_percent_ratio']:.1%})")
    print(f"30%以内: 115個 (23.0%) → {error_stats['within_30_percent']}個 ({error_stats['within_30_percent_ratio']:.1%})")
    print(f"50%以内: 150個 (30.0%) → {error_stats['within_50_percent']}個 ({error_stats['within_50_percent_ratio']:.1%})")

    # EVAL形式の新しい評価指標を表示
    print(f"\n=== 評価指標（EVAL形式） ===")
    print(f"[EVAL] mean(key_mean_abs_err_div_pred_overall) across material_key = {error_stats['eval_mean']:.6f} (n_keys={error_stats['eval_total_keys']})")
    print(f"[EVAL] median(key_mean_abs_err_div_pred_overall) across material_key = {error_stats['eval_median']:.6f} (n_keys={error_stats['eval_total_keys']})")
    print(f"[EVAL] #keys within 20%: {error_stats['eval_within_20']} / {error_stats['eval_total_keys']} ({error_stats['eval_within_20_ratio']*100:.2f}%)")
    print(f"[EVAL] #keys within 30%: {error_stats['eval_within_30']} / {error_stats['eval_total_keys']} ({error_stats['eval_within_30_ratio']*100:.2f}%)")
    print(f"[EVAL] #keys within 50%: {error_stats['eval_within_50']} / {error_stats['eval_total_keys']} ({error_stats['eval_within_50_ratio']*100:.2f}%)")

    # 誤差が大きいMaterial Keyの例を表示
    df_key_total_sorted = df_key_total.copy()
    df_key_total_sorted['abs_error_rate_total'] = np.abs(df_key_total_sorted['error_rate_total'])
    df_key_total_sorted = df_key_total_sorted.sort_values('abs_error_rate_total', ascending=False)

    print(f"\n=== 予測誤差が大きいMaterial Key TOP10 ===")
    print(df_key_total_sorted[['material_key', 'actual', 'predicted', 'error_rate_total', 'count_actual_positive_total']].head(10).to_string(index=False))

    print(f"\n=== 予測精度が高いMaterial Key TOP10 ===")
    df_key_total_sorted_asc = df_key_total_sorted[df_key_total_sorted['abs_error_rate_total'].notna()].sort_values('abs_error_rate_total')
    print(df_key_total_sorted_asc[['material_key', 'actual', 'predicted', 'error_rate_total', 'count_actual_positive_total']].head(10).to_string(index=False))

    return model, importance, {'rmse': rmse, 'mae': mae, 'error_stats': error_stats}, (df_results, df_key_date, df_key_total)

def main():
    """メイン処理"""

    print("="*50)
    print("学習・予測処理開始（予測誤差分析機能付き）")
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

    # 日付範囲
    if 'file_date' in df_features.columns:
        df_features['file_date'] = pd.to_datetime(df_features['file_date'], errors='coerce')
        date_min = df_features['file_date'].min()
        date_max = df_features['file_date'].max()
        print(f"日付範囲: {date_min} 〜 {date_max}")

    # 2. モデル学習とテストデータでの予測・評価
    print(f"\n2. モデル学習とテストデータ評価開始...")

    model, importance, metrics, error_analysis = train_and_predict_with_test_period(
        df_features,
        test_start='2025-01-01',  # 6ヶ月（1月〜6月）
        test_end='2025-06-30',
        is_optuna=True  # Optunaありで実行
    )

    if model is None:
        print("エラー: モデル学習に失敗しました")
        return None

    # 3. 結果を保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    date_dir = datetime.now().strftime('%Y%m%d')  # YYYYMMDD形式の日付ディレクトリ

    # S3に保存
    print("\nS3に結果を保存中...")
    import joblib

    # モデルをバイト列にシリアライズ
    model_buffer = BytesIO()
    joblib.dump(model, model_buffer)
    model_buffer.seek(0)

    # S3にモデルをアップロード（日付ディレクトリ付き）
    model_key = f"models/{date_dir}/model_with_error_{timestamp}.pkl"
    s3.put_object(
        Bucket=bucket_name,
        Key=model_key,
        Body=model_buffer.getvalue()
    )
    print(f"S3モデル保存: s3://{bucket_name}/{model_key}")

    # 最新版としても保存
    model_latest_key = "models/model_with_error_latest.pkl"
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
    importance_latest_key = "models/importance_with_error_latest.parquet"
    s3.put_object(
        Bucket=bucket_name,
        Key=importance_latest_key,
        Body=importance_buffer.getvalue()
    )
    print(f"S3最新特徴量重要度保存: s3://{bucket_name}/{importance_latest_key}")

    # メトリクスをS3に保存（error_statsを含む完全版）
    metrics_full = metrics.copy()
    if 'error_stats' in metrics_full:
        # error_statsを展開してフラットな構造にする
        error_stats = metrics_full.pop('error_stats')
        # int64などのNumPy型をPython標準型に変換
        for key, value in error_stats.items():
            if hasattr(value, 'item'):
                error_stats[key] = value.item()
        metrics_full.update(error_stats)

    # NumPy型をPython標準型に変換
    for key, value in metrics_full.items():
        if hasattr(value, 'item'):
            metrics_full[key] = value.item()

    metrics_buffer = BytesIO()
    import json
    metrics_json = json.dumps(metrics_full, indent=2)
    metrics_buffer.write(metrics_json.encode('utf-8'))
    metrics_buffer.seek(0)
    metrics_key = f"models/{date_dir}/metrics_with_error_{timestamp}.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=metrics_key,
        Body=metrics_buffer.getvalue(),
        ContentType='application/json'
    )
    print(f"S3評価指標保存: s3://{bucket_name}/{metrics_key}")

    # 最新版としても保存
    metrics_latest_key = "models/metrics_with_error_latest.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=metrics_latest_key,
        Body=metrics_buffer.getvalue(),
        ContentType='application/json'
    )
    print(f"S3最新評価指標保存: s3://{bucket_name}/{metrics_latest_key}")

    # エラー統計サマリーも別途保存
    if 'error_stats' in metrics:
        test_start = '2025-01-01'  # テスト期間の開始
        test_end = '2025-06-30'    # テスト期間の終了
        summary_stats = {
            'timestamp': timestamp,
            'test_period': f"{test_start} to {test_end}",
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'total_materials': metrics['error_stats']['total_materials'],
            'within_20_percent': metrics['error_stats']['within_20_percent'],
            'within_20_percent_ratio': metrics['error_stats']['within_20_percent_ratio'],
            'within_30_percent': metrics['error_stats']['within_30_percent'],
            'within_30_percent_ratio': metrics['error_stats']['within_30_percent_ratio'],
            'within_50_percent': metrics['error_stats']['within_50_percent'],
            'within_50_percent_ratio': metrics['error_stats']['within_50_percent_ratio'],
            'error_mean': metrics['error_stats']['error_mean'],
            'error_median': metrics['error_stats']['error_median'],
            'error_std': metrics['error_stats']['error_std']
        }

        summary_buffer = BytesIO()
        summary_json = json.dumps(summary_stats, indent=2)
        summary_buffer.write(summary_json.encode('utf-8'))
        summary_buffer.seek(0)
        summary_key = f"models/{date_dir}/error_summary_{timestamp}.json"
        s3.put_object(
            Bucket=bucket_name,
            Key=summary_key,
            Body=summary_buffer.getvalue(),
            ContentType='application/json'
        )
        print(f"S3エラー統計サマリー保存: s3://{bucket_name}/{summary_key}")

        # 最新版としても保存
        summary_latest_key = "models/error_summary_latest.json"
        s3.put_object(
            Bucket=bucket_name,
            Key=summary_latest_key,
            Body=summary_buffer.getvalue(),
            ContentType='application/json'
        )
        print(f"S3最新エラー統計サマリー保存: s3://{bucket_name}/{summary_latest_key}")

    # 誤差分析結果をS3に保存
    if error_analysis:
        df_results, df_key_date, df_key_total = error_analysis

        # Material Key × File Date毎の結果
        key_date_buffer = BytesIO()
        df_key_date.to_parquet(key_date_buffer, index=False)
        key_date_buffer.seek(0)  # バッファの先頭に戻す
        key_date_key = f"models/{date_dir}/error_analysis_key_date_{timestamp}.parquet"
        s3.put_object(
            Bucket=bucket_name,
            Key=key_date_key,
            Body=key_date_buffer.getvalue(),
        )
        print(f"S3誤差分析(Key×Date)保存: s3://{bucket_name}/{key_date_key}")

        # 最新版としても保存
        key_date_latest_key = "models/error_analysis_key_date_latest.parquet"
        s3.put_object(
            Bucket=bucket_name,
            Key=key_date_latest_key,
            Body=key_date_buffer.getvalue(),
        )
        print(f"S3最新誤差分析(Key×Date)保存: s3://{bucket_name}/{key_date_latest_key}")

        # Material Key毎の結果（6ヶ月全体）
        key_total_buffer = BytesIO()
        df_key_total.to_parquet(key_total_buffer, index=False)
        key_total_buffer.seek(0)  # バッファの先頭に戻す
        key_total_key = f"models/{date_dir}/error_analysis_key_total_{timestamp}.parquet"
        s3.put_object(
            Bucket=bucket_name,
            Key=key_total_key,
            Body=key_total_buffer.getvalue(),
        )
        print(f"S3誤差分析(Key全体)保存: s3://{bucket_name}/{key_total_key}")

        # 最新版としても保存
        key_total_latest_key = "models/error_analysis_key_total_latest.parquet"
        s3.put_object(
            Bucket=bucket_name,
            Key=key_total_latest_key,
            Body=key_total_buffer.getvalue(),
        )
        print(f"S3最新誤差分析(Key全体)保存: s3://{bucket_name}/{key_total_latest_key}")

        # 詳細結果
        results_buffer = BytesIO()
        df_results.to_parquet(results_buffer)
        results_key = f"models/{date_dir}/prediction_results_{timestamp}.parquet"
        s3.put_object(
            Bucket=bucket_name,
            Key=results_key,
            Body=results_buffer.getvalue()
        )
        print(f"S3予測結果詳細保存: s3://{bucket_name}/{results_key}")

        # 最新版としても保存
        results_latest_key = "models/prediction_results_latest.parquet"
        s3.put_object(
            Bucket=bucket_name,
            Key=results_latest_key,
            Body=results_buffer.getvalue()
        )
        print(f"S3最新予測結果詳細保存: s3://{bucket_name}/{results_latest_key}")

    # ローカル保存は削除（S3のみに保存）

    print("\n" + "="*50)
    print("処理完了！")
    print("="*50)

    return model, importance, metrics, error_analysis

if __name__ == "__main__":
    main()