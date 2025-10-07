#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ヤマサ確定注文需要予測システム - 学習・予測
README.md必須要件準拠版
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
import gc
import warnings
import os
import json
import boto3
from smart_open import open as smart_open
from dotenv import load_dotenv
import argparse
warnings.filterwarnings('ignore')

# 環境変数を読み込み
load_dotenv()

def create_features_enhanced(df):
    """強化版特徴量作成（全曜日特徴量）"""
    print("  強化版特徴量作成中...")

    df = df.sort_values(['material_key', 'file_date'])

    # 基本的な日付特徴量
    df['day_of_week_f'] = df['file_date'].dt.dayofweek.astype('float32')
    df['month_f'] = df['month'].astype('float32')
    df['week_number_f'] = df['week_number'].astype('float32')

    # material_key × 曜日の過去平均（既存）
    print("    material_dow_mean_f作成中...")
    df['material_dow_mean_f'] = df.groupby(['material_key', 'day_of_week_f'])['actual_value'].transform(
        lambda x: x.expanding().mean().shift(1).fillna(0)
    ).astype('float32')

    # 基本的なラグ特徴量
    for lag in [1, 7]:
        print(f"    lag_{lag}_f作成中...")
        df[f'lag_{lag}_f'] = df.groupby('material_key')['actual_value'].shift(lag).fillna(0).astype('float32')

    # 移動平均
    print("    rolling_mean_7_f作成中...")
    df['rolling_mean_7_f'] = df.groupby('material_key')['actual_value'].shift(1).transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    ).fillna(0).astype('float32')

    # 追加の曜日特徴量（新規）
    if 'product_key' in df.columns:
        print("    product_dow_mean_f作成中...")
        df['product_dow_mean_f'] = df.groupby(['product_key', 'day_of_week_f'])['actual_value'].transform(
            lambda x: x.expanding().mean().shift(1).fillna(0)
        ).astype('float32')

    if 'store_code' in df.columns:
        print("    store_dow_mean_f作成中...")
        df['store_dow_mean_f'] = df.groupby(['store_code', 'day_of_week_f'])['actual_value'].transform(
            lambda x: x.expanding().mean().shift(1).fillna(0)
        ).astype('float32')

    if 'usage_type' in df.columns:
        print("    usage_dow_mean_f作成中...")
        df['usage_dow_mean_f'] = df.groupby(['usage_type', 'day_of_week_f'])['actual_value'].transform(
            lambda x: x.expanding().mean().shift(1).fillna(0)
        ).astype('float32')

    return df

def calculate_evaluation_metrics(df_test, y_true, y_pred):
    """README.md必須要件に従った評価指標計算"""

    # 基本指標
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    # material_key毎の誤差率計算
    result_df = pd.DataFrame({
        'material_key': df_test['material_key'].values,
        'actual': y_true,
        'predicted': y_pred,
        'file_date': df_test['file_date'].values
    })

    # 誤差率: (予測値-実績値の絶対値)/実績値
    result_df['error_rate'] = np.where(
        result_df['actual'] != 0,
        np.abs(result_df['predicted'] - result_df['actual']) / result_df['actual'],
        np.nan
    )

    # material_key毎の平均誤差率
    mk_error_rates = result_df.groupby('material_key')['error_rate'].apply(
        lambda x: x.dropna().mean() if len(x.dropna()) > 0 else np.nan
    ).reset_index()
    mk_error_rates.columns = ['material_key', 'avg_error_rate']

    # 有効なmaterial_keyのみ
    valid_mk = mk_error_rates[mk_error_rates['avg_error_rate'].notna()]
    total_mk = len(valid_mk)

    if total_mk > 0:
        # 誤差率統計
        error_rates = valid_mk['avg_error_rate']
        avg_error_rate = error_rates.mean()
        median_error_rate = error_rates.median()

        # 閾値内のmaterial_key数・割合
        within_20_count = (error_rates <= 0.20).sum()
        within_30_count = (error_rates <= 0.30).sum()
        within_50_count = (error_rates <= 0.50).sum()

        within_20_pct = within_20_count / total_mk * 100
        within_30_pct = within_30_count / total_mk * 100
        within_50_pct = within_50_count / total_mk * 100
    else:
        avg_error_rate = median_error_rate = np.nan
        within_20_count = within_30_count = within_50_count = 0
        within_20_pct = within_30_pct = within_50_pct = 0

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        '誤差率平均値': avg_error_rate,
        '誤差率中央値': median_error_rate,
        '20%以内_件数': within_20_count,
        '20%以内_割合': within_20_pct,
        '30%以内_件数': within_30_count,
        '30%以内_割合': within_30_pct,
        '50%以内_件数': within_50_count,
        '50%以内_割合': within_50_pct,
        '総material_key数': total_mk
    }

    return metrics, result_df, mk_error_rates

def train_model_and_evaluate(df_features, train_end_date):
    """モデル学習と評価（README.md要件準拠）"""

    train_end_dt = pd.to_datetime(train_end_date)

    # 学習データ: train_end_dateまで
    train_mask = df_features['file_date'] <= train_end_dt
    df_train = df_features[train_mask]

    # 評価用データ: 2025/1-6（既存データがある場合のみ）
    eval_mask = (df_features['file_date'] >= '2025-01-01') & (df_features['file_date'] <= '2025-06-30')
    df_eval = df_features[eval_mask]

    print(f"    学習期間: ~{train_end_dt.date()}")
    print(f"    学習データ: {len(df_train):,}行")
    print(f"    評価データ: {len(df_eval):,}行（既存データがある場合）")

    # 特徴量とターゲット
    feature_cols = [col for col in df_features.columns if col.endswith('_f')]
    print(f"    特徴量数: {len(feature_cols)}")

    X_train = df_train[feature_cols].values
    y_train = df_train['actual_value'].values

    # LightGBMモデル
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'num_threads': 4,
        'force_col_wise': True,
        'random_state': 42
    }

    print("    モデル学習中...")
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100, callbacks=[lgb.log_evaluation(0)])

    # 特徴量重要度
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    # 評価データがある場合は評価実行
    evaluation_metrics = None
    if len(df_eval) > 0:
        print("    評価データで精度検証中...")
        X_eval = df_eval[feature_cols].values
        y_eval = df_eval['actual_value'].values
        y_pred_eval = model.predict(X_eval)

        evaluation_metrics, _, _ = calculate_evaluation_metrics(df_eval, y_eval, y_pred_eval)

        print("\n    【評価結果（既存データでの検証）】")
        print(f"    RMSE: {evaluation_metrics['RMSE']:.2f}")
        print(f"    MAE: {evaluation_metrics['MAE']:.2f}")
        print(f"    誤差率平均値: {evaluation_metrics['誤差率平均値']:.2%}")
        print(f"    誤差率中央値: {evaluation_metrics['誤差率中央値']:.2%}")
        print(f"    20%以内: {evaluation_metrics['20%以内_件数']}個 ({evaluation_metrics['20%以内_割合']:.1f}%)")
        print(f"    30%以内: {evaluation_metrics['30%以内_件数']}個 ({evaluation_metrics['30%以内_割合']:.1f}%)")
        print(f"    50%以内: {evaluation_metrics['50%以内_件数']}個 ({evaluation_metrics['50%以内_割合']:.1f}%)")

    # メモリ解放
    del X_train, y_train, df_train
    if len(df_eval) > 0:
        del X_eval, y_eval, df_eval
    gc.collect()

    return model, importance, feature_cols, evaluation_metrics

def predict_future_periods(df_features, model, feature_cols, train_end_date, step_count):
    """step_count月分の将来予測（README.md要件準拠）"""

    train_end_dt = pd.to_datetime(train_end_date)

    # 学習期間のデータから最新の特徴量状態を取得
    train_data = df_features[df_features['file_date'] <= train_end_dt].copy()

    # 実績値がゼロより大きい値があり、かつ十分な活動があるmaterial_keyのみを予測対象とする
    print("    実績値がゼロより大きく、活動が十分なmaterial_keyを特定中...")

    # 各material_keyの統計を計算
    mk_stats = train_data.groupby('material_key').agg({
        'actual_value': ['count', 'sum', 'mean', lambda x: (x > 0).sum()]
    }).reset_index()
    mk_stats.columns = ['material_key', 'total_records', 'total_value', 'avg_value', 'positive_days']

    # フィルタ条件: 実績値合計 > 0 かつ 実績値がある日が5日以上
    valid_material_keys = mk_stats[
        (mk_stats['total_value'] > 0) &
        (mk_stats['positive_days'] >= 5)
    ]['material_key'].values

    print(f"    フィルタ後のmaterial_key数: {len(valid_material_keys):,} (実績合計>0 かつ 実績日数≥5日)")

    # 予測対象を絞り込み
    train_data_filtered = train_data[train_data['material_key'].isin(valid_material_keys)].copy()

    # 各material_keyの最新の特徴量状態
    latest_features = train_data_filtered.groupby('material_key').tail(1)[
        ['material_key', 'store_code', 'product_key', 'usage_type'] + feature_cols
    ].reset_index(drop=True)

    print(f"    予測対象material_key数: {len(latest_features):,} (実績値>0のみ)")

    predictions = []

    for step in range(1, step_count + 1):
        # 予測対象月
        pred_date = train_end_dt + pd.DateOffset(months=step)
        pred_year_month = pred_date.strftime('%Y-%m')

        print(f"    Step {step}: {pred_year_month} の予測中...")

        # その月の各日に対して予測
        month_start = pred_date.replace(day=1)
        if step < step_count:
            month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
        else:
            # 最終月は要件に応じて調整（ここでは月末まで）
            month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)

        # その月の全日
        date_range = pd.date_range(month_start, month_end, freq='D')

        monthly_predictions = []

        for date in date_range:
            # その日の特徴量を更新
            pred_features = latest_features.copy()
            pred_features['day_of_week_f'] = date.dayofweek
            pred_features['month_f'] = date.month
            pred_features['week_number_f'] = date.isocalendar().week

            # 時系列特徴量を適切に更新（ラグ特徴量は最後の実績値を使用）
            # 注意: 実際の予測では前日の予測値を使うべきだが、
            # 現在は簡単化のため最新の実績値ベースの特徴量を使用

            # 曜日別過去平均特徴量は日付変化で影響を受ける可能性があるが、
            # 過去データベースなので基本的には変わらない

            # 予測実行
            X_pred = pred_features[feature_cols].values
            daily_pred = model.predict(X_pred)

            # 結果格納
            daily_result = pd.DataFrame({
                'material_key': pred_features['material_key'],
                'prediction_date': date,
                'predicted_value': daily_pred,
                'step': step,
                'year_month': pred_year_month
            })

            monthly_predictions.append(daily_result)

        # その月の予測結果をまとめる
        if monthly_predictions:
            month_df = pd.concat(monthly_predictions, ignore_index=True)
            predictions.append(month_df)

    # 全ての予測結果を結合
    if predictions:
        final_predictions = pd.concat(predictions, ignore_index=True)

        # material_key × 月の集計
        monthly_summary = final_predictions.groupby(['material_key', 'year_month']).agg({
            'predicted_value': ['sum', 'mean', 'count'],
            'step': 'first'
        }).reset_index()

        monthly_summary.columns = [
            'material_key', 'year_month', 'monthly_total', 'daily_average', 'prediction_days', 'step'
        ]

        print(f"    予測完了: {len(final_predictions):,}日分の予測")
        print(f"    月次サマリー: {len(monthly_summary):,}レコード")

        return final_predictions, monthly_summary
    else:
        return pd.DataFrame(), pd.DataFrame()

def enhance_predictions_with_actuals(daily_predictions, df_features):
    """予測結果にactual_valueと誤差情報を追加"""

    print("    予測結果にactual_value等を追加中...")

    # actual_valueを追加（既存データがある場合）
    enhanced_predictions = daily_predictions.copy()

    # 予測日付に対応するactual_valueを取得
    df_actuals = df_features[['material_key', 'file_date', 'actual_value']].copy()
    df_actuals['file_date'] = pd.to_datetime(df_actuals['file_date'])

    # 予測結果とマージ
    enhanced_predictions['prediction_date'] = pd.to_datetime(enhanced_predictions['prediction_date'])
    enhanced_predictions = enhanced_predictions.merge(
        df_actuals,
        left_on=['material_key', 'prediction_date'],
        right_on=['material_key', 'file_date'],
        how='left'
    ).drop('file_date', axis=1)

    # actual_valueがない場合はNaNのまま残す
    print(f"    actual_valueがある予測行: {enhanced_predictions['actual_value'].notna().sum():,}行")

    # 誤差計算（actual_valueがある場合のみ）
    enhanced_predictions['daily_error'] = np.where(
        enhanced_predictions['actual_value'].notna(),
        np.abs(enhanced_predictions['predicted_value'] - enhanced_predictions['actual_value']),
        np.nan
    )

    # 予測日別の平均誤差（その日の全material_keyの平均）
    daily_avg_errors = enhanced_predictions.groupby('prediction_date')['daily_error'].mean().reset_index()
    daily_avg_errors.columns = ['prediction_date', 'daily_avg_error']

    enhanced_predictions = enhanced_predictions.merge(daily_avg_errors, on='prediction_date', how='left')

    # material_key別の全期間平均誤差
    mk_avg_errors = enhanced_predictions.groupby('material_key')['daily_error'].mean().reset_index()
    mk_avg_errors.columns = ['material_key', 'material_key_avg_error']

    enhanced_predictions = enhanced_predictions.merge(mk_avg_errors, on='material_key', how='left')

    # 不要な中間カラムを削除
    enhanced_predictions = enhanced_predictions.drop('daily_error', axis=1)

    print(f"    拡張予測結果: {len(enhanced_predictions):,}行")

    return enhanced_predictions

def save_results_to_s3(daily_predictions, monthly_summary, importance, model, evaluation_metrics, timestamp, train_end_date, step_count, df_features=None):
    """結果をS3に保存（README.md構造準拠）"""

    bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', 'fiby-yamasa-prediction')

    print(f"\nS3への保存開始: s3://{bucket_name}/output_data/predictions/")

    # 予測結果を拡張（actual_value等を追加）
    if df_features is not None:
        enhanced_predictions = enhance_predictions_with_actuals(daily_predictions, df_features)
    else:
        enhanced_predictions = daily_predictions

    # 1. 予測結果（predictions_*.parquet）- 拡張版
    predictions_key = f'output_data/predictions/predictions_{timestamp}.parquet'
    predictions_path = f's3://{bucket_name}/{predictions_key}'

    with smart_open(predictions_path, 'wb') as f:
        enhanced_predictions.to_parquet(f, index=False)
    print(f"  ✓ 予測結果（拡張版）: {predictions_path}")

    # 最新版として_latestサフィックス付きでも保存（README.md要件）
    predictions_latest_key = f'output_data/predictions/predictions_latest.parquet'
    predictions_latest_path = f's3://{bucket_name}/{predictions_latest_key}'

    with smart_open(predictions_latest_path, 'wb') as f:
        enhanced_predictions.to_parquet(f, index=False)
    print(f"  ✓ 予測結果（最新版）: {predictions_latest_path}")

    # 2. material_key別エラー率（既存データがある場合のみ）
    if evaluation_metrics:
        # 評価結果をCSVで保存（カラム名を英語に変更）
        bykey_errors_key = f'output_data/predictions/bykey_errors_{timestamp}.csv'
        bykey_errors_path = f's3://{bucket_name}/{bykey_errors_key}'

        # カラム名を英語に変換
        eval_summary = pd.DataFrame([evaluation_metrics])
        eval_summary_en = eval_summary.rename(columns={
            '誤差率平均値': 'avg_error_rate',
            '誤差率中央値': 'median_error_rate',
            '20%以内_件数': 'within_20pct_count',
            '20%以内_割合': 'within_20pct_ratio',
            '30%以内_件数': 'within_30pct_count',
            '30%以内_割合': 'within_30pct_ratio',
            '50%以内_件数': 'within_50pct_count',
            '50%以内_割合': 'within_50pct_ratio',
            '総material_key数': 'total_material_keys'
        })

        with smart_open(bykey_errors_path, 'w') as f:
            eval_summary_en.to_csv(f, index=False)
        print(f"  ✓ エラー率: {bykey_errors_path}")

    # 3. 特徴量重要度（feature_importance_*.csv）- Tableau読み込み対応、全件出力
    importance_key = f'output_data/predictions/feature_importance_{timestamp}.csv'
    importance_path = f's3://{bucket_name}/{importance_key}'

    # 全ての特徴量を出力（上位だけでなく全件）
    with smart_open(importance_path, 'w') as f:
        # ヘッダー行を明示的に制御してTableau読み込み対応
        importance.to_csv(f, index=False, header=True)
    print(f"  ✓ 特徴量重要度: {importance_path}")

    # 4. パラメータ（params_*.json）
    params_data = {
        'train_end_date': train_end_date,
        'step_count': step_count,
        'timestamp': timestamp,
        'model_type': 'LightGBM',
        'feature_count': len(importance)
    }

    params_key = f'output_data/predictions/params_{timestamp}.json'
    params_path = f's3://{bucket_name}/{params_key}'

    with smart_open(params_path, 'w') as f:
        json.dump(params_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ パラメータ: {params_path}")

    # 5. 評価メトリクス（metrics_*.json）
    if evaluation_metrics:
        metrics_key = f'output_data/predictions/metrics_{timestamp}.json'
        metrics_path = f's3://{bucket_name}/{metrics_key}'

        # NaN値をNoneに変換し、カラム名を英語に変換
        clean_metrics = {}
        column_mapping = {
            '誤差率平均値': 'avg_error_rate',
            '誤差率中央値': 'median_error_rate',
            '20%以内_件数': 'within_20pct_count',
            '20%以内_割合': 'within_20pct_ratio',
            '30%以内_件数': 'within_30pct_count',
            '30%以内_割合': 'within_30pct_ratio',
            '50%以内_件数': 'within_50pct_count',
            '50%以内_割合': 'within_50pct_ratio',
            '総material_key数': 'total_material_keys'
        }

        for k, v in evaluation_metrics.items():
            # カラム名を英語に変換
            key = column_mapping.get(k, k)
            if pd.isna(v):
                clean_metrics[key] = None
            else:
                clean_metrics[key] = float(v) if isinstance(v, np.number) else v

        with smart_open(metrics_path, 'w') as f:
            json.dump(clean_metrics, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 評価メトリクス: {metrics_path}")

    print(f"\nS3保存完了: README.md構造に従って保存しました")

    return {
        'predictions': predictions_path,
        'importance': importance_path,
        'params': params_path
    }

def main():
    """メイン処理（README.md要件準拠）"""

    parser = argparse.ArgumentParser(description='ヤマサ確定注文需要予測 - 学習・予測')
    parser.add_argument('--train-end-date', default='2024-12-31', help='学習データの終了日')
    parser.add_argument('--step-count', type=int, default=6, help='予測ステップ数（月単位）')

    args = parser.parse_args()

    print("="*90)
    print(" ヤマサ確定注文需要予測システム - 学習・予測（README.md要件準拠）")
    print("="*90)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_end_date = args.train_end_date
    step_count = args.step_count

    print(f"実行パラメータ:")
    print(f"  学習終了日: {train_end_date}")
    print(f"  予測ステップ数: {step_count}ヶ月")
    print(f"  予測期間: 2025/01 ~ 2025/{step_count:02d}")

    # データ読み込み
    print("\nデータ読み込み中...")
    df = pd.read_parquet('data/df_confirmed_order_input_yamasa_fill_zero.parquet')

    print(f"元データサイズ: {df.shape}")

    # 必要な期間のデータのみ
    df['file_date'] = pd.to_datetime(df['file_date'])
    train_end_dt = pd.to_datetime(train_end_date)
    data_start_dt = train_end_dt - pd.DateOffset(months=12)  # 学習用に1年分
    data_end_dt = pd.to_datetime("2025-06-30")  # 評価用データ期間まで

    df_filtered = df[(df['file_date'] >= data_start_dt) & (df['file_date'] <= data_end_dt)].copy()
    del df
    gc.collect()

    print(f"フィルタリング後: {df_filtered.shape}")
    print(f"データ期間: {df_filtered['file_date'].min().date()} ~ {df_filtered['file_date'].max().date()}")

    # メモリ最適化
    for col in df_filtered.columns:
        if df_filtered[col].dtype == 'object' and col != 'file_date':
            df_filtered[col] = df_filtered[col].astype('category')
        elif df_filtered[col].dtype == 'float64':
            df_filtered[col] = df_filtered[col].astype('float32')

    # 特徴量作成
    print("\n" + "-"*80)
    print("特徴量作成")
    print("-"*80)

    df_enhanced = create_features_enhanced(df_filtered.copy())

    # モデル学習と評価
    print("\n" + "-"*80)
    print("モデル学習・評価")
    print("-"*80)

    model, importance, feature_cols, evaluation_metrics = train_model_and_evaluate(df_enhanced, train_end_date)

    # 将来予測
    print("\n" + "-"*80)
    print(f"将来予測（step_count={step_count}）")
    print("-"*80)

    daily_predictions, monthly_summary = predict_future_periods(
        df_enhanced, model, feature_cols, train_end_date, step_count
    )

    # df_enhancedは後でS3保存時に使用するため、まだ削除しない
    del df_filtered
    gc.collect()

    # 結果表示
    print("\n" + "="*90)
    print(" 実行結果サマリー")
    print("="*90)

    if not daily_predictions.empty:
        print(f"予測対象material_key数: {daily_predictions['material_key'].nunique():,}")
        print(f"総予測レコード数: {len(daily_predictions):,}")
        print(f"予測期間: {daily_predictions['prediction_date'].min().date()} ~ {daily_predictions['prediction_date'].max().date()}")

        print("\n月別予測統計:")
        if not monthly_summary.empty:
            month_stats = monthly_summary.groupby('year_month').agg({
                'material_key': 'count',
                'monthly_total': ['min', 'max', 'mean'],
                'prediction_days': 'first'
            }).round(2)
            print(month_stats)

    # 曜日特徴量の重要度
    dow_features = importance[importance['feature'].str.contains('dow', case=False)]
    if not dow_features.empty:
        dow_importance_pct = dow_features['importance'].sum() / importance['importance'].sum() * 100
        print(f"\n曜日特徴量の重要度割合: {dow_importance_pct:.1f}%")

        print("\n主要な曜日特徴量:")
        for idx, row in dow_features.head(5).iterrows():
            print(f"  - {row['feature']:<25} : {row['importance']:>15.0f}")

    # S3に保存
    print("\n" + "="*90)
    print(" S3への結果保存")
    print("="*90)

    if not daily_predictions.empty:
        s3_paths = save_results_to_s3(
            daily_predictions, monthly_summary, importance, model,
            evaluation_metrics, timestamp, train_end_date, step_count, df_enhanced
        )

        print("\n" + "="*90)
        print(" 処理完了")
        print("="*90)
        print(f"実行タイムスタンプ: {timestamp}")
        print(f"README.md要件に従った予測結果がS3に保存されました")

        return s3_paths, daily_predictions, monthly_summary
    else:
        print("\n予測結果が空のため、保存をスキップします")
        return None, None, None

if __name__ == "__main__":
    s3_paths, daily_predictions, monthly_summary = main()