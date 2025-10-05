#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""学習・予測実行スクリプト（予測誤差分析機能付き）"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from datetime import datetime
from dotenv import load_dotenv

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

    return df_results, df_key_date, df_key_total

def analyze_error_distribution(df_key_total):
    """誤差率の分布を分析

    Args:
        df_key_total: Material Key毎の集計結果

    Returns:
        誤差率別の統計
    """

    # 絶対誤差率で判定
    df_valid = df_key_total[~df_key_total['error_rate_total'].isna()].copy()
    df_valid['abs_error_rate_total'] = np.abs(df_valid['error_rate_total'])

    total_materials = len(df_valid)

    # 誤差率別のカウント
    within_20 = (df_valid['abs_error_rate_total'] <= 0.2).sum()
    within_30 = (df_valid['abs_error_rate_total'] <= 0.3).sum()
    within_50 = (df_valid['abs_error_rate_total'] <= 0.5).sum()

    stats = {
        'total_materials': total_materials,
        'within_20_percent': within_20,
        'within_30_percent': within_30,
        'within_50_percent': within_50,
        'within_20_percent_ratio': within_20 / total_materials if total_materials > 0 else 0,
        'within_30_percent_ratio': within_30 / total_materials if total_materials > 0 else 0,
        'within_50_percent_ratio': within_50 / total_materials if total_materials > 0 else 0
    }

    return stats

def train_and_predict_with_test_period(df_features, test_start='2025-01-01', test_end='2025-06-30', target_col='actual_value'):
    """指定期間をテストデータとしてモデル学習と予測を実行"""

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

    # 検証用データを学習データから分割
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

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

    print(f"\n=== Material Key毎の予測精度分布 ===")
    print(f"分析対象Material Key数: {error_stats['total_materials']}")
    print(f"予測誤差20%以内: {error_stats['within_20_percent']}個 ({error_stats['within_20_percent_ratio']:.1%})")
    print(f"予測誤差30%以内: {error_stats['within_30_percent']}個 ({error_stats['within_30_percent_ratio']:.1%})")
    print(f"予測誤差50%以内: {error_stats['within_50_percent']}個 ({error_stats['within_50_percent_ratio']:.1%})")

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

    # ローカルの特徴量ファイルを読み込む
    features_file = "output_data/features/df_features_yamasa_latest.parquet"

    if not os.path.exists(features_file):
        print(f"エラー: 特徴量ファイルが見つかりません: {features_file}")
        print("先に create_features_local.py を実行してください")
        return None

    print(f"\n1. 特徴量データ読込中...")
    print(f"ファイル: {features_file}")

    df_features = pd.read_parquet(features_file)
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

    # 2. モデル学習とテストデータでの予測・評価
    print(f"\n2. モデル学習とテストデータ評価開始...")

    model, importance, metrics, error_analysis = train_and_predict_with_test_period(
        df_features,
        test_start='2025-01-01',
        test_end='2025-06-30'
    )

    if model is None:
        print("エラー: モデル学習に失敗しました")
        return None

    # 3. 結果を保存
    output_dir = "output_data/models"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # モデルを保存
    import joblib
    model_file = f"{output_dir}/model_with_error_{timestamp}.pkl"
    joblib.dump(model, model_file)
    print(f"\nモデル保存: {model_file}")

    # 特徴量重要度を保存
    importance_file = f"{output_dir}/importance_{timestamp}.csv"
    importance.to_csv(importance_file, index=False)
    print(f"特徴量重要度保存: {importance_file}")

    # メトリクスを保存
    metrics_file = f"{output_dir}/metrics_with_error_{timestamp}.json"
    pd.Series({k: v for k, v in metrics.items() if k != 'error_stats'}).to_json(metrics_file)
    print(f"評価指標保存: {metrics_file}")

    # 誤差分析結果を保存
    if error_analysis:
        df_results, df_key_date, df_key_total = error_analysis

        # Material Key × File Date毎の結果
        key_date_file = f"{output_dir}/error_analysis_key_date_{timestamp}.csv"
        df_key_date.to_csv(key_date_file, index=False)
        print(f"誤差分析結果（Key×Date）保存: {key_date_file}")

        # Material Key毎の結果（6ヶ月全体）
        key_total_file = f"{output_dir}/error_analysis_key_total_{timestamp}.csv"
        df_key_total.to_csv(key_total_file, index=False)
        print(f"誤差分析結果（Key全体）保存: {key_total_file}")

        # 詳細結果
        results_file = f"{output_dir}/prediction_results_{timestamp}.parquet"
        df_results.to_parquet(results_file)
        print(f"予測結果詳細保存: {results_file}")

    print("\n" + "="*50)
    print("処理完了！")
    print("="*50)

    return model, importance, metrics, error_analysis

if __name__ == "__main__":
    main()