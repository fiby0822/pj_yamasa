#!/usr/bin/env python3
"""
パターン1の詳細分析
統合モデルとusage_type別モデルの比較を含む
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# サンプリング設定
SAMPLE_SIZE = 500000
RANDOM_STATE = 42

# 指定された12個の特徴量
SPECIFIED_FEATURES = [
    'year_f', 'month_f', 'lag_1_f', 'lag_2_f', 'lag_3_f',
    'rolling_mean_2_f', 'rolling_mean_3_f', 'rolling_mean_6_f',
    'cumulative_mean_2_f', 'cumulative_mean_3_f', 'cumulative_mean_6_f', 'cumulative_mean_12_f'
]

def create_features(df):
    """12個の特徴量を作成"""
    # カラム名の正規化
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # 日付と数値の処理
    df['file_date'] = pd.to_datetime(df['file_date'], errors='coerce')
    df['actual_value'] = pd.to_numeric(df['actual_value'], errors='coerce').fillna(0)

    # ソート
    df = df.sort_values(['material_key', 'file_date'])

    features = []

    # 日付特徴量
    df['year_f'] = df['file_date'].dt.year
    df['month_f'] = df['file_date'].dt.month
    features.extend(['year_f', 'month_f'])

    # material_keyごとの時系列特徴量
    df_grouped = df.groupby('material_key')['actual_value']

    # Lag特徴量
    for lag in [1, 2, 3]:
        df[f'lag_{lag}_f'] = df_grouped.shift(lag).fillna(0)
        features.append(f'lag_{lag}_f')

    # Rolling Mean
    for window in [2, 3, 6]:
        df[f'rolling_mean_{window}_f'] = df_grouped.transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        ).fillna(0)
        features.append(f'rolling_mean_{window}_f')

    # Cumulative Mean
    for window in [2, 3, 6, 12]:
        df[f'cumulative_mean_{window}_f'] = df_grouped.transform(
            lambda x: x.expanding(min_periods=min(window, 1)).mean()
        ).fillna(0)
        features.append(f'cumulative_mean_{window}_f')

    return df, features

def calculate_error_metrics(y_true, y_pred):
    """詳細な誤差メトリクスを計算"""
    # 基本メトリクス
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 誤差率の計算（ゼロ除算を回避）
    with np.errstate(divide='ignore', invalid='ignore'):
        error_rate = np.abs(y_true - y_pred) / np.where(y_true != 0, np.abs(y_true), 1) * 100
        error_rate = np.where(y_true == 0,
                             np.where(y_pred == 0, 0, 100),  # 実際が0で予測も0なら誤差率0%、予測が0でなければ100%
                             error_rate)

    # 平均誤差率と中央誤差率
    mean_error_rate = np.mean(error_rate)
    median_error_rate = np.median(error_rate)

    # 誤差率別の集計（material_keyごと）
    error_within_20 = np.sum(error_rate <= 20)
    error_within_30 = np.sum(error_rate <= 30)
    error_within_50 = np.sum(error_rate <= 50)

    total_count = len(y_true)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mean_error_rate': mean_error_rate,
        'median_error_rate': median_error_rate,
        'within_20': error_within_20,
        'within_20_pct': error_within_20 / total_count * 100,
        'within_30': error_within_30,
        'within_30_pct': error_within_30 / total_count * 100,
        'within_50': error_within_50,
        'within_50_pct': error_within_50 / total_count * 100,
        'total_samples': total_count
    }

    return metrics

def train_and_evaluate(X_train, y_train, X_test, y_test, model_name="Model"):
    """モデルを学習して評価"""
    # 固定パラメータ
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # 学習
    model.fit(X_train, y_train)

    # 予測
    y_pred = model.predict(X_test)

    # メトリクス計算
    metrics = calculate_error_metrics(y_test, y_pred)
    metrics['model_name'] = model_name

    return model, metrics, y_pred

def main():
    print("="*80)
    print("パターン1 詳細分析: 統合モデル vs usage_type別モデル")
    print("="*80)

    # データ読み込み
    print("\nデータ読込中...")
    df = pd.read_parquet('s3://fiby-yamasa-prediction/data/df_confirmed_order_input_yamasa_fill_zero.parquet')
    print(f"読込完了: {len(df):,} 行")

    # サンプリング
    if len(df) > SAMPLE_SIZE:
        print(f"サンプリング実施: {SAMPLE_SIZE:,} 行")
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    # 特徴量作成
    print("\n特徴量作成中...")
    df, feature_cols = create_features(df)
    print(f"作成された特徴量: {len(feature_cols)}個")

    # データ準備
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df['actual_value']

    # データ分割（時系列なのでシャッフル無し）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=False
    )

    # データフレームとして保持（usage_type情報のため）
    df_train = df.iloc[:len(X_train)].copy()
    df_test = df.iloc[len(X_train):].copy()

    print(f"\nデータセット:")
    print(f"  訓練データ: {len(X_train):,} 行")
    print(f"  テストデータ: {len(X_test):,} 行")

    results = []

    # ===== 1. 統合モデル =====
    print("\n" + "="*60)
    print("1. 統合モデル（全データで学習）")
    print("="*60)

    model_all, metrics_all, y_pred_all = train_and_evaluate(
        X_train, y_train, X_test, y_test, "統合モデル"
    )

    print(f"\nテストデータの評価:")
    print(f"  RMSE: {metrics_all['rmse']:.2f}")
    print(f"  MAE: {metrics_all['mae']:.2f}")
    print(f"  R²: {metrics_all['r2']:.4f}")
    print(f"  平均誤差率: {metrics_all['mean_error_rate']:.2f}%")
    print(f"  中央誤差率: {metrics_all['median_error_rate']:.2f}%")
    print(f"  20%以内: {metrics_all['within_20']:,}個 ({metrics_all['within_20_pct']:.1f}%)")
    print(f"  30%以内: {metrics_all['within_30']:,}個 ({metrics_all['within_30_pct']:.1f}%)")
    print(f"  50%以内: {metrics_all['within_50']:,}個 ({metrics_all['within_50_pct']:.1f}%)")

    results.append(metrics_all)

    # ===== 2. usage_type別モデル =====
    print("\n" + "="*60)
    print("2. usage_type別モデル")
    print("="*60)

    # usage_typeがあるか確認
    if 'usage_type' in df.columns:
        # usage_typeごとのデータ分割
        usage_types = df['usage_type'].unique()
        print(f"\nusage_type種類: {usage_types}")

        # 各usage_typeのモデル結果を保存
        usage_predictions = []
        usage_actuals = []

        for usage_type in usage_types:
            # 訓練データとテストデータをフィルタリング
            train_mask = df_train['usage_type'] == usage_type
            test_mask = df_test['usage_type'] == usage_type

            X_train_usage = X_train[train_mask]
            y_train_usage = y_train[train_mask]
            X_test_usage = X_test[test_mask]
            y_test_usage = y_test[test_mask]

            if len(X_train_usage) > 0 and len(X_test_usage) > 0:
                print(f"\n{usage_type}モデル:")
                print(f"  訓練: {len(X_train_usage):,} 行")
                print(f"  テスト: {len(X_test_usage):,} 行")

                # モデル学習と評価
                model_usage, metrics_usage, y_pred_usage = train_and_evaluate(
                    X_train_usage, y_train_usage, X_test_usage, y_test_usage,
                    f"{usage_type}モデル"
                )

                print(f"  RMSE: {metrics_usage['rmse']:.2f}")
                print(f"  MAE: {metrics_usage['mae']:.2f}")

                # 予測結果を統合
                usage_predictions.extend(y_pred_usage)
                usage_actuals.extend(y_test_usage)

        # 統合した予測結果で全体評価
        if usage_predictions:
            print("\n" + "-"*60)
            print("usage_type別モデル（統合評価）:")

            y_pred_combined = np.array(usage_predictions)
            y_test_combined = np.array(usage_actuals)

            metrics_combined = calculate_error_metrics(y_test_combined, y_pred_combined)
            metrics_combined['model_name'] = "usage_type別モデル"

            print(f"  RMSE: {metrics_combined['rmse']:.2f}")
            print(f"  MAE: {metrics_combined['mae']:.2f}")
            print(f"  R²: {metrics_combined['r2']:.4f}")
            print(f"  平均誤差率: {metrics_combined['mean_error_rate']:.2f}%")
            print(f"  中央誤差率: {metrics_combined['median_error_rate']:.2f}%")
            print(f"  20%以内: {metrics_combined['within_20']:,}個 ({metrics_combined['within_20_pct']:.1f}%)")
            print(f"  30%以内: {metrics_combined['within_30']:,}個 ({metrics_combined['within_30_pct']:.1f}%)")
            print(f"  50%以内: {metrics_combined['within_50']:,}個 ({metrics_combined['within_50_pct']:.1f}%)")

            results.append(metrics_combined)
    else:
        print("usage_typeカラムが見つかりません。統合モデルのみ評価。")

    # ===== 3. 比較結果 =====
    if len(results) == 2:
        print("\n" + "="*80)
        print("比較結果サマリ")
        print("="*80)

        print("\n| 指標 | 統合モデル | usage_type別モデル | 改善幅 |")
        print("|------|-----------|------------------|--------|")

        # RMSE
        rmse_diff = results[0]['rmse'] - results[1]['rmse']
        print(f"| RMSE | {results[0]['rmse']:.2f} | {results[1]['rmse']:.2f} | "
              f"{rmse_diff:+.2f} {'✅' if rmse_diff > 0 else '❌'} |")

        # MAE
        mae_diff = results[0]['mae'] - results[1]['mae']
        print(f"| MAE | {results[0]['mae']:.2f} | {results[1]['mae']:.2f} | "
              f"{mae_diff:+.2f} {'✅' if mae_diff > 0 else '❌'} |")

        # 平均誤差率
        mean_err_diff = results[0]['mean_error_rate'] - results[1]['mean_error_rate']
        print(f"| 平均誤差率 | {results[0]['mean_error_rate']:.2f}% | "
              f"{results[1]['mean_error_rate']:.2f}% | "
              f"{mean_err_diff:+.2f}% {'✅' if mean_err_diff > 0 else '❌'} |")

        # 中央誤差率
        med_err_diff = results[0]['median_error_rate'] - results[1]['median_error_rate']
        print(f"| 中央誤差率 | {results[0]['median_error_rate']:.2f}% | "
              f"{results[1]['median_error_rate']:.2f}% | "
              f"{med_err_diff:+.2f}% {'✅' if med_err_diff > 0 else '❌'} |")

        # 20%以内
        within20_diff = results[1]['within_20'] - results[0]['within_20']
        within20_pct_diff = results[1]['within_20_pct'] - results[0]['within_20_pct']
        print(f"| 20%以内 | {results[0]['within_20']:,}個 ({results[0]['within_20_pct']:.1f}%) | "
              f"{results[1]['within_20']:,}個 ({results[1]['within_20_pct']:.1f}%) | "
              f"+{within20_diff:,}個 (+{within20_pct_diff:.1f}%) {'✅' if within20_diff > 0 else '❌'} |")

        # 30%以内
        within30_diff = results[1]['within_30'] - results[0]['within_30']
        within30_pct_diff = results[1]['within_30_pct'] - results[0]['within_30_pct']
        print(f"| 30%以内 | {results[0]['within_30']:,}個 ({results[0]['within_30_pct']:.1f}%) | "
              f"{results[1]['within_30']:,}個 ({results[1]['within_30_pct']:.1f}%) | "
              f"+{within30_diff:,}個 (+{within30_pct_diff:.1f}%) {'✅' if within30_diff > 0 else '❌'} |")

        # 50%以内
        within50_diff = results[1]['within_50'] - results[0]['within_50']
        within50_pct_diff = results[1]['within_50_pct'] - results[0]['within_50_pct']
        print(f"| 50%以内 | {results[0]['within_50']:,}個 ({results[0]['within_50_pct']:.1f}%) | "
              f"{results[1]['within_50']:,}個 ({results[1]['within_50_pct']:.1f}%) | "
              f"+{within50_diff:,}個 (+{within50_pct_diff:.1f}%) {'✅' if within50_diff > 0 else '❌'} |")

    # 結果をCSV保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df_results = pd.DataFrame(results)
    output_file = f"pattern1_detailed_analysis_{timestamp}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n結果をCSVに保存: {output_file}")

    print("\n" + "="*80)
    print("分析完了")
    print("="*80)

if __name__ == "__main__":
    main()