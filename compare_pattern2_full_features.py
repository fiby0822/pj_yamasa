#!/usr/bin/env python3
"""
パターン2: フル特徴量版（現在実装している全特徴量、Optuna有り）
ゼロ補完済みデータから読み込み
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import boto3
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optunaのインストール確認
try:
    import optuna
except ImportError:
    os.system('pip install optuna')
    import optuna

# サンプリング設定
SAMPLE_SIZE = 500000  # 50万行でテスト
RANDOM_STATE = 42

def create_full_features(df):
    """フル特徴量を作成（現在の実装を再現）"""
    print("="*50)
    print("特徴量作成（フル版）")
    print("="*50)

    # カラム名の正規化
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # 日付処理
    if 'file_date' in df.columns:
        df['file_date'] = pd.to_datetime(df['file_date'], errors='coerce')

    # 数値変換
    if 'actual_value' in df.columns:
        df['actual_value'] = pd.to_numeric(df['actual_value'], errors='coerce').fillna(0)

    # ソート
    if 'material_key' in df.columns and 'file_date' in df.columns:
        df = df.sort_values(['material_key', 'file_date'])

    features = []

    # 1. 日付特徴量（包括的）
    if 'file_date' in df.columns:
        print("日付特徴量作成中...")
        df['year_f'] = df['file_date'].dt.year
        df['month_f'] = df['file_date'].dt.month
        df['day_f'] = df['file_date'].dt.day
        df['day_of_week_f'] = df['file_date'].dt.dayofweek
        df['week_of_year_f'] = df['file_date'].dt.isocalendar().week.astype(int)
        df['quarter_f'] = df['file_date'].dt.quarter
        df['is_month_end_f'] = df['file_date'].dt.is_month_end.astype(int)
        df['is_month_start_f'] = df['file_date'].dt.is_month_start.astype(int)

        # 営業日フラグ
        is_weekend = df['file_date'].dt.dayofweek.isin([5, 6])
        is_year_end = ((df['file_date'].dt.month == 12) &
                       (df['file_date'].dt.day.isin([30, 31])))
        df['is_business_day_f'] = (~is_weekend & ~is_year_end).astype(int)

        # その他の日付特徴量
        df['days_in_month_f'] = df['file_date'].dt.days_in_month
        df['days_since_month_start_f'] = df['file_date'].dt.day - 1
        df['days_until_month_end_f'] = df['days_in_month_f'] - df['file_date'].dt.day
        df['is_first_half_year_f'] = (df['file_date'].dt.month <= 6).astype(int)
        df['is_first_week_of_month_f'] = (df['file_date'].dt.day <= 7).astype(int)
        df['is_last_week_of_month_f'] = (df['file_date'].dt.day > (df['days_in_month_f'] - 7)).astype(int)

        date_features = ['year_f', 'month_f', 'day_f', 'day_of_week_f', 'week_of_year_f',
                        'quarter_f', 'is_month_end_f', 'is_month_start_f', 'is_business_day_f',
                        'days_in_month_f', 'days_since_month_start_f', 'days_until_month_end_f',
                        'is_first_half_year_f', 'is_first_week_of_month_f', 'is_last_week_of_month_f']
        features.extend(date_features)

    # 2. 時系列特徴量（複数のグループレベル）
    print("時系列特徴量作成中...")

    # material_keyごとの特徴量
    if 'material_key' in df.columns and 'actual_value' in df.columns:
        df_grouped = df.groupby('material_key')['actual_value']

        # Lag特徴量
        for lag in [1, 2, 3, 7]:
            feature_name = f'material_lag_{lag}_f'
            df[feature_name] = df_grouped.shift(lag).fillna(0)
            features.append(feature_name)

        # Rolling統計量
        for window in [3, 5, 7, 14, 21, 28]:
            # Mean
            feature_name = f'material_rolling_mean_{window}_f'
            df[feature_name] = df_grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            ).fillna(0)
            features.append(feature_name)

            # Std
            if window <= 7:
                feature_name = f'material_rolling_std_{window}_f'
                df[feature_name] = df_grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                ).fillna(0)
                features.append(feature_name)

            # Max/Min
            if window <= 7:
                feature_name = f'material_rolling_max_{window}_f'
                df[feature_name] = df_grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                ).fillna(0)
                features.append(feature_name)

                feature_name = f'material_rolling_min_{window}_f'
                df[feature_name] = df_grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                ).fillna(0)
                features.append(feature_name)

        # Cumulative Mean
        for window in [2, 3, 4, 5, 6, 9, 12]:
            feature_name = f'material_cumulative_mean_{window}_f'
            df[feature_name] = df_grouped.transform(
                lambda x: x.expanding(min_periods=min(window, 1)).mean()
            ).fillna(0)
            features.append(feature_name)

    # store_codeごとの特徴量
    if 'store_code' in df.columns and 'actual_value' in df.columns:
        df_grouped = df.groupby('store_code')['actual_value']

        # 基本統計量
        df['store_mean_f'] = df_grouped.transform('mean').fillna(0)
        df['store_std_f'] = df_grouped.transform('std').fillna(0)
        df['store_max_f'] = df_grouped.transform('max').fillna(0)
        df['store_min_f'] = df_grouped.transform('min').fillna(0)

        features.extend(['store_mean_f', 'store_std_f', 'store_max_f', 'store_min_f'])

        # Rolling特徴量
        for window in [7, 14]:
            feature_name = f'store_rolling_mean_{window}_f'
            df[feature_name] = df_grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            ).fillna(0)
            features.append(feature_name)

    # product_keyごとの特徴量
    if 'product_key' in df.columns and 'actual_value' in df.columns:
        df_grouped = df.groupby('product_key')['actual_value']

        df['product_mean_f'] = df_grouped.transform('mean').fillna(0)
        df['product_std_f'] = df_grouped.transform('std').fillna(0)

        features.extend(['product_mean_f', 'product_std_f'])

    # カテゴリごとの特徴量
    for cat_col in ['category_lvl_1', 'category_lvl_2', 'category_lvl_3']:
        if cat_col in df.columns and 'actual_value' in df.columns:
            df_grouped = df.groupby(cat_col)['actual_value']

            feature_name = f'{cat_col}_mean_f'
            df[feature_name] = df_grouped.transform('mean').fillna(0)
            features.append(feature_name)

            feature_name = f'{cat_col}_std_f'
            df[feature_name] = df_grouped.transform('std').fillna(0)
            features.append(feature_name)

    # 作成された特徴量を確認
    created_features = [f for f in features if f in df.columns]

    print(f"\n作成された特徴量: {len(created_features)}個")

    return df, created_features

def train_model_with_optuna(X_train, y_train, X_test, y_test):
    """Optunaでハイパーパラメータ最適化を行いながらモデルを学習"""
    print("\n" + "="*50)
    print("モデル学習（Optuna有り）")
    print("="*50)

    def objective(trial):
        """Optuna用の目的関数"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }

        # モデル学習
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # 検証データで評価
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return rmse

    # Optuna最適化
    print("ハイパーパラメータ最適化中...")
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print(f"\n最適パラメータ:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 最適パラメータでモデル再学習
    print("\n最適パラメータでモデル再学習中...")
    best_params = study.best_params
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = -1

    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)

    # 予測
    print("予測中...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 評価
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    print("\n訓練データの評価:")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  MAE: {train_mae:.2f}")
    print(f"  R2: {train_r2:.4f}")

    print("\nテストデータの評価:")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE: {test_mae:.2f}")
    print(f"  R2: {test_r2:.4f}")

    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n特徴量重要度（上位10）:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    metrics = {
        'pattern': 'パターン2（フル特徴量、Optuna有り）',
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'n_features': len(X_train.columns),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'best_params': str(study.best_params)
    }

    return model, metrics, feature_importance

def main():
    print("="*70)
    print("パターン2: フル特徴量版（全特徴量、Optuna有り）")
    print("="*70)

    # S3設定
    bucket_name = "fiby-yamasa-prediction"
    input_key = "data/df_confirmed_order_input_yamasa_fill_zero.parquet"

    print(f"\nデータ読込中: s3://{bucket_name}/{input_key}")

    # データ読み込み
    df = pd.read_parquet(f"s3://{bucket_name}/{input_key}")
    print(f"読込完了: {len(df):,} 行")

    # サンプリング
    if len(df) > SAMPLE_SIZE:
        print(f"\nサンプリング実施: {SAMPLE_SIZE:,} 行")
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    # 特徴量作成
    df, feature_cols = create_full_features(df)

    # データ準備
    X = df[feature_cols]
    y = df['actual_value']

    # 無限大と欠損値の処理
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    # データ分割
    print(f"\nデータ分割中...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=False
    )

    print(f"  訓練データ: {len(X_train):,} 行")
    print(f"  テストデータ: {len(X_test):,} 行")
    print(f"  特徴量数: {len(feature_cols)} 個")

    # モデル学習と評価
    model, metrics, importance = train_model_with_optuna(X_train, y_train, X_test, y_test)

    # 結果保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # メトリクスをCSV保存
    metrics_df = pd.DataFrame([metrics])
    metrics_file = f"pattern2_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\nメトリクス保存: {metrics_file}")

    # 特徴量重要度をCSV保存
    importance_file = f"pattern2_importance_{timestamp}.csv"
    importance.to_csv(importance_file, index=False)
    print(f"特徴量重要度保存: {importance_file}")

    print("\n" + "="*70)
    print("パターン2 完了")
    print("="*70)

    return metrics

if __name__ == "__main__":
    metrics = main()