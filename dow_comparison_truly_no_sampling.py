#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""曜日別過去平均特徴量の比較分析（完全にサンプリングなし版）"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import gc
import warnings
warnings.filterwarnings('ignore')

def calculate_material_key_metrics(df_test, y_true, y_pred):
    """material_key毎の評価指標を計算"""

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # material_key毎の誤差率計算
    result_df = pd.DataFrame({
        'material_key': df_test['material_key'].values,
        'actual': y_true,
        'predicted': y_pred
    })

    # 誤差率計算
    result_df['error_pct'] = np.where(
        result_df['actual'] != 0,
        np.abs((result_df['actual'] - result_df['predicted']) / result_df['actual']) * 100,
        np.nan
    )

    # 全体の平均・中央誤差率
    valid_errors = result_df['error_pct'].dropna()
    mean_pct_error = valid_errors.mean() if len(valid_errors) > 0 else np.nan
    median_pct_error = valid_errors.median() if len(valid_errors) > 0 else np.nan

    # material_key毎の平均誤差率
    mk_metrics = result_df.groupby('material_key')['error_pct'].apply(
        lambda x: x.dropna().mean() if len(x.dropna()) > 0 else np.nan
    ).reset_index()
    mk_metrics.columns = ['material_key', 'mean_error_pct']

    # 有効なmaterial_keyのみ
    valid_mk = mk_metrics[mk_metrics['mean_error_pct'].notna()]
    total_mk = len(valid_mk)

    if total_mk > 0:
        within_20 = (valid_mk['mean_error_pct'] <= 20).sum()
        within_30 = (valid_mk['mean_error_pct'] <= 30).sum()
        within_50 = (valid_mk['mean_error_pct'] <= 50).sum()

        within_20_pct = within_20 / total_mk * 100
        within_30_pct = within_30 / total_mk * 100
        within_50_pct = within_50 / total_mk * 100
    else:
        within_20 = within_30 = within_50 = 0
        within_20_pct = within_30_pct = within_50_pct = 0

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        '平均誤差率': mean_pct_error,
        '中央誤差率': median_pct_error,
        '20%以内_件数': within_20,
        '20%以内_割合': within_20_pct,
        '30%以内_件数': within_30,
        '30%以内_割合': within_30_pct,
        '50%以内_件数': within_50,
        '50%以内_割合': within_50_pct,
        '総material_key数': total_mk
    }

    print(f"    テストデータ行数: {len(result_df):,}")
    print(f"    material_key数: {total_mk:,}")
    if total_mk > 0:
        print(f"    誤差率分布: 最小{valid_mk['mean_error_pct'].min():.1f}% "
              f"中央{valid_mk['mean_error_pct'].median():.1f}% "
              f"最大{valid_mk['mean_error_pct'].max():.1f}%")

    return metrics

def format_comparison_table(results_basic, results_enhanced):
    """比較結果をテーブル形式で表示"""

    improvements = {}
    for key in ['RMSE', 'MAE', '平均誤差率', '中央誤差率']:
        improvements[key] = results_basic[key] - results_enhanced[key]

    for key in ['20%以内_割合', '30%以内_割合', '50%以内_割合']:
        improvements[key] = results_enhanced[key] - results_basic[key]

    print("\n" + "="*85)
    print(" 比較結果（完全サンプリングなし・全material_key）")
    print("="*85)
    print(f"| {'指標':<12} | {'基本モデル':>18} | {'強化版モデル':>18} | {'改善幅':>25} |")
    print(f"|{'-'*14}|{'-'*20}|{'-'*20}|{'-'*27}|")

    # 各指標の表示
    for metric, label in [('RMSE', 'RMSE'), ('MAE', 'MAE'),
                          ('平均誤差率', '平均誤差率'), ('中央誤差率', '中央誤差率')]:
        check = "✅" if improvements[metric] > 0 else ""
        if '率' in metric:
            print(f"| {label:<12} | {results_basic[metric]:>17.2f}% | {results_enhanced[metric]:>17.2f}% | {improvements[metric]:>+12.2f}% {check:>10} |")
        else:
            print(f"| {label:<12} | {results_basic[metric]:>18.2f} | {results_enhanced[metric]:>18.2f} | {improvements[metric]:>+13.2f} {check:>10} |")

    # 20%/30%/50%以内
    for threshold in [20, 30, 50]:
        key = f'{threshold}%以内'
        check = "✅" if improvements[f'{key}_割合'] > 0 else ""
        basic_str = f"{results_basic[f'{key}_件数']}個 ({results_basic[f'{key}_割合']:.1f}%)"
        enhanced_str = f"{results_enhanced[f'{key}_件数']}個 ({results_enhanced[f'{key}_割合']:.1f}%)"
        diff = results_enhanced[f'{key}_件数'] - results_basic[f'{key}_件数']
        sign = "+" if diff >= 0 else ""
        diff_str = f"{sign}{diff}個 ({sign}{improvements[f'{key}_割合']:.1f}%)"
        print(f"| {key:<12} | {basic_str:>18} | {enhanced_str:>18} | {diff_str:>25} {check} |")

    print("="*85)
    print(f"\n※ 評価対象material_key数: {results_basic['総material_key数']:,}")
    print(f"※ テストデータ期間: 2025/01/01 ~ 2025/06/30")
    print(f"※ サンプリング: 一切なし（全データ）")

    improved_count = sum(1 for key in ['RMSE', 'MAE', '平均誤差率', '中央誤差率', '20%以内_割合', '30%以内_割合', '50%以内_割合']
                        if (key in improvements and
                            ((key in ['RMSE', 'MAE', '平均誤差率', '中央誤差率'] and improvements[key] > 0) or
                             (key.endswith('_割合') and improvements[key] > 0))))

    print(f"\n改善された指標: {improved_count}/7")
    return improvements

def create_features_basic(df):
    """基本特徴量作成（軽量版）"""
    print("  基本特徴量作成中...")

    df = df.sort_values(['material_key', 'file_date'])

    # 最小限の日付特徴量
    df['day_of_week_f'] = df['file_date'].dt.dayofweek.astype('float32')
    df['month_f'] = df['month'].astype('float32')

    # material_key × 曜日の過去平均（チャンクサイズを使って効率化）
    print("    material_dow_mean_f作成中...")
    df['material_dow_mean_f'] = df.groupby(['material_key', 'day_of_week_f'])['actual_value'].transform(
        lambda x: x.expanding().mean().shift(1).fillna(0)
    ).astype('float32')

    # 最小限のラグ特徴量
    for lag in [1, 7]:
        print(f"    lag_{lag}_f作成中...")
        df[f'lag_{lag}_f'] = df.groupby('material_key')['actual_value'].shift(lag).fillna(0).astype('float32')

    # 1つの移動平均のみ
    print("    rolling_mean_7_f作成中...")
    df['rolling_mean_7_f'] = df.groupby('material_key')['actual_value'].shift(1).transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    ).fillna(0).astype('float32')

    return df

def create_features_enhanced(df):
    """強化版特徴量作成（軽量版）"""
    print("  強化版特徴量作成中...")

    # 基本特徴量
    df = create_features_basic(df)

    # 追加の曜日特徴量（効率化）
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

def train_and_evaluate(df_features, train_end_date, model_name):
    """学習と評価"""

    train_end_dt = pd.to_datetime(train_end_date)
    test_start_dt = pd.to_datetime("2025-01-01")
    test_end_dt = pd.to_datetime("2025-06-30")

    # データ分割
    train_mask = df_features['file_date'] <= train_end_dt
    test_mask = (df_features['file_date'] >= test_start_dt) & (df_features['file_date'] <= test_end_dt)

    df_train = df_features[train_mask]
    df_test = df_features[test_mask]

    print(f"    学習期間: ~{train_end_dt.date()}")
    print(f"    テスト期間: {test_start_dt.date()} ~ {test_end_dt.date()}")
    print(f"    学習: {len(df_train):,}行, テスト: {len(df_test):,}行")

    if len(df_test) == 0:
        print("    警告: テストデータが空")
        return None, None

    # 特徴量とターゲット
    feature_cols = [col for col in df_features.columns if col.endswith('_f')]
    print(f"    特徴量数: {len(feature_cols)}")

    X_train = df_train[feature_cols].values
    y_train = df_train['actual_value'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['actual_value'].values

    # LightGBM学習（軽量設定）
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'verbose': -1,
        'num_threads': 4,
        'force_col_wise': True,
        'max_bin': 127  # メモリ削減
    }

    print("    モデル学習中...")
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=50, callbacks=[lgb.log_evaluation(0)])  # 学習回数を削減

    # 予測と評価
    print("    予測中...")
    y_pred = model.predict(X_test)

    print("    評価指標計算中...")
    metrics = calculate_material_key_metrics(df_test, y_test, y_pred)
    metrics['model'] = model_name

    # 特徴量重要度
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    dow_features = importance[importance['feature'].str.contains('dow', case=False)]
    metrics['dow_features_count'] = len(dow_features)
    metrics['dow_importance_pct'] = dow_features['importance'].sum() / importance['importance'].sum() * 100 if importance['importance'].sum() > 0 else 0

    return metrics, importance

def main():
    """メイン処理"""
    print("="*85)
    print(" 曜日特徴量比較分析（完全サンプリングなし版）")
    print("="*85)

    train_end_date = "2024-12-31"

    # データ読み込み
    print("\nデータ読み込み中...")
    df = pd.read_parquet('data/df_confirmed_order_input_yamasa_fill_zero.parquet')

    print(f"元データサイズ: {df.shape}")

    # 期間フィルタリング（メモリ節約のため）
    df['file_date'] = pd.to_datetime(df['file_date'])
    train_end_dt = pd.to_datetime(train_end_date)
    data_start_dt = train_end_dt - pd.DateOffset(months=4)  # 4ヶ月前からに縮小
    data_end_dt = pd.to_datetime("2025-06-30")

    df_filtered = df[(df['file_date'] >= data_start_dt) & (df['file_date'] <= data_end_dt)].copy()
    del df
    gc.collect()

    print(f"期間フィルタリング後: {df_filtered.shape}")
    print(f"データ期間: {df_filtered['file_date'].min().date()} ~ {df_filtered['file_date'].max().date()}")

    # テスト期間にデータがあるmaterial_keyのみ選択（サンプリングなし）
    test_mask = (df_filtered['file_date'] >= '2025-01-01') & (df_filtered['file_date'] <= '2025-06-30')
    test_material_keys = df_filtered[test_mask]['material_key'].unique()

    # ※重要: サンプリングは一切行わない
    df_final = df_filtered[df_filtered['material_key'].isin(test_material_keys)].copy()
    del df_filtered
    gc.collect()

    print(f"最終データサイズ: {df_final.shape}")

    # メモリ最適化
    for col in df_final.columns:
        if df_final[col].dtype == 'object' and col != 'file_date':
            df_final[col] = df_final[col].astype('category')
        elif df_final[col].dtype == 'float64':
            df_final[col] = df_final[col].astype('float32')

    # テストデータの最終確認
    test_mask = (df_final['file_date'] >= '2025-01-01') & (df_final['file_date'] <= '2025-06-30')
    test_count = test_mask.sum()
    test_mk_count = df_final[test_mask]['material_key'].nunique()

    print(f"テスト期間(2025/1~6)データ: {test_count:,}行")
    print(f"テスト期間のmaterial_key数: {test_mk_count:,}")
    print(f"全material_key数: {df_final['material_key'].nunique():,}")

    # 1. 基本特徴量での学習
    print("\n" + "-"*70)
    print("1. 基本モデル（material_dow_meanのみ）の学習")
    print("-"*70)

    df_basic = create_features_basic(df_final.copy())
    results_basic, importance_basic = train_and_evaluate(df_basic, train_end_date, "基本モデル")
    del df_basic
    gc.collect()

    # 2. 強化版特徴量での学習
    print("\n" + "-"*70)
    print("2. 強化版モデル（全DOW特徴量）の学習")
    print("-"*70)

    df_enhanced = create_features_enhanced(df_final.copy())
    results_enhanced, importance_enhanced = train_and_evaluate(df_enhanced, train_end_date, "強化版モデル")
    del df_enhanced
    gc.collect()

    # 結果比較
    improvements = format_comparison_table(results_basic, results_enhanced)

    # 新規追加特徴量の重要度
    print("\n" + "="*85)
    print(" 新規追加された曜日特徴量の重要度")
    print("="*85)

    new_features = ['product_dow_mean_f', 'store_dow_mean_f', 'usage_dow_mean_f']
    new_importance = importance_enhanced[importance_enhanced['feature'].isin(new_features)]
    if not new_importance.empty:
        print("\n新規追加された曜日特徴量:")
        for idx, row in new_importance.iterrows():
            print(f"  - {row['feature']:<20} : {row['importance']:>15.0f}")
        print(f"  合計重要度: {new_importance['importance'].sum():>15.0f}")

    # 結果をCSVに保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_df = pd.DataFrame({
        'モデル': ['基本モデル', '強化版モデル'],
        'RMSE': [results_basic['RMSE'], results_enhanced['RMSE']],
        'MAE': [results_basic['MAE'], results_enhanced['MAE']],
        '平均誤差率': [results_basic['平均誤差率'], results_enhanced['平均誤差率']],
        '中央誤差率': [results_basic['中央誤差率'], results_enhanced['中央誤差率']],
        '20%以内_割合': [results_basic['20%以内_割合'], results_enhanced['20%以内_割合']],
        '30%以内_割合': [results_basic['30%以内_割合'], results_enhanced['30%以内_割合']],
        '50%以内_割合': [results_basic['50%以内_割合'], results_enhanced['50%以内_割合']],
    })

    summary_df.to_csv(f'dow_comparison_truly_no_sampling_{timestamp}.csv', index=False)

    print("\n" + "="*85)
    print(" 分析完了（完全サンプリングなし版）")
    print("="*85)
    print(f"結果をCSVに保存: dow_comparison_truly_no_sampling_{timestamp}.csv")

    return summary_df

if __name__ == "__main__":
    results = main()