#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""学習・予測実行スクリプト（可視化機能付き）"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# 環境変数を読み込み
load_dotenv()

# プロットスタイルの設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def visualize_data_distribution(df, target_col='Actual Value', output_dir='output_data/visualizations'):
    """データ分布の可視化"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. ヒストグラム（生データ）
    axes[0, 0].hist(df[target_col].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution of Actual Value', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Actual Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. ログ変換後のヒストグラム
    log_values = np.log1p(df[target_col].clip(lower=0)).dropna()
    axes[0, 1].hist(log_values, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Distribution of log1p(Actual Value)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('log1p(Actual Value)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 時系列プロット
    if 'File Date' in df.columns:
        df_sorted = df.sort_values('File Date')
        daily_sum = df_sorted.groupby('File Date')[target_col].sum()
        axes[1, 0].plot(daily_sum.index, daily_sum.values, color='green', alpha=0.7)
        axes[1, 0].set_title('Daily Total of Actual Value', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Total Actual Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

    # 4. 箱ひげ図（月別）
    if 'month' in df.columns:
        monthly_data = [df[df['month'] == m][target_col].dropna() for m in sorted(df['month'].unique())]
        axes[1, 1].boxplot(monthly_data, labels=sorted(df['month'].unique()))
        axes[1, 1].set_title('Monthly Distribution of Actual Value', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Actual Value')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/data_distribution.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("📊 データ分布の可視化完了")

def visualize_seasonal_patterns(df, target_col='Actual Value', output_dir='output_data/visualizations'):
    """季節性パターンの可視化"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 月別平均
    if 'month' in df.columns:
        monthly_avg = df.groupby('month')[target_col].mean()
        axes[0, 0].bar(monthly_avg.index, monthly_avg.values, color='steelblue')
        axes[0, 0].set_title('Average by Month', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Average Actual Value')
        axes[0, 0].grid(True, alpha=0.3)

    # 2. 曜日別平均
    if 'day_of_week' in df.columns:
        weekday_avg = df.groupby('day_of_week')[target_col].mean()
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 1].bar(weekday_avg.index, weekday_avg.values, color='coral')
        axes[0, 1].set_title('Average by Weekday', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Average Actual Value')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(weekday_names)
        axes[0, 1].grid(True, alpha=0.3)

    # 3. 日別平均（1-31日）
    if 'day' in df.columns:
        daily_avg = df.groupby('day')[target_col].mean()
        axes[1, 0].plot(daily_avg.index, daily_avg.values, marker='o', color='green')
        axes[1, 0].set_title('Average by Day of Month', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Average Actual Value')
        axes[1, 0].grid(True, alpha=0.3)

    # 4. 年別トレンド
    if 'year' in df.columns:
        yearly_sum = df.groupby('year')[target_col].sum()
        axes[1, 1].bar(yearly_sum.index, yearly_sum.values, color='purple')
        axes[1, 1].set_title('Yearly Total', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Total Actual Value')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/seasonal_patterns.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("📊 季節性パターンの可視化完了")

def visualize_model_performance(y_true, y_pred, model_name="LightGBM", output_dir='output_data/visualizations'):
    """モデル性能の可視化"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. 実測値 vs 予測値
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Actual Value')
    axes[0, 0].set_ylabel('Predicted Value')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 残差プロット
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Predicted Value')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 残差のヒストグラム
    axes[0, 2].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 2].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Residuals')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Q-Qプロット
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 予測誤差の分布
    percentage_error = np.abs(residuals) / (np.abs(y_true) + 1e-10) * 100
    axes[1, 1].hist(percentage_error, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Distribution of Percentage Error', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Percentage Error (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(x=np.median(percentage_error), color='r', linestyle='--', lw=2, label=f'Median: {np.median(percentage_error):.2f}%')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. 累積誤差
    cumulative_error = np.cumsum(residuals)
    axes[1, 2].plot(cumulative_error, color='blue')
    axes[1, 2].set_title('Cumulative Error', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Sample Index')
    axes[1, 2].set_ylabel('Cumulative Error')
    axes[1, 2].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_performance.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("📊 モデル性能の可視化完了")

def visualize_feature_importance(importance_df, top_n=20, output_dir='output_data/visualizations'):
    """特徴量重要度の可視化"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Top N特徴量を取得
    top_features = importance_df.nlargest(top_n, 'importance')

    # 1. 棒グラフ
    axes[0].barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'].values)
    axes[0].set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Importance')
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()

    # 2. 累積重要度
    importance_sorted = importance_df.sort_values('importance', ascending=False)
    cumsum = importance_sorted['importance'].cumsum() / importance_sorted['importance'].sum() * 100

    axes[1].plot(range(1, len(cumsum) + 1), cumsum.values, marker='o', color='green')
    axes[1].set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Features')
    axes[1].set_ylabel('Cumulative Importance (%)')
    axes[1].axhline(y=80, color='r', linestyle='--', lw=2, label='80% threshold')
    axes[1].axhline(y=90, color='orange', linestyle='--', lw=2, label='90% threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("📊 特徴量重要度の可視化完了")

def print_evaluation_metrics(y_true, y_pred, model_name="Model"):
    """評価指標の詳細表示"""

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPEの計算（0除算を回避）
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.inf

    # 中央値絶対誤差
    median_ae = np.median(np.abs(y_true - y_pred))

    # 誤差の標準偏差
    std_error = np.std(y_true - y_pred)

    print("\n" + "="*60)
    print(f"📊 {model_name} - 評価指標サマリー")
    print("="*60)
    print(f"✅ RMSE (Root Mean Square Error):     {rmse:,.4f}")
    print(f"✅ MAE (Mean Absolute Error):          {mae:,.4f}")
    print(f"✅ R² Score:                           {r2:.4f}")
    print(f"✅ MAPE (Mean Absolute % Error):       {mape:.2f}%")
    print(f"✅ Median Absolute Error:              {median_ae:,.4f}")
    print(f"✅ Standard Deviation of Error:        {std_error:,.4f}")
    print("-"*60)

    # 統計サマリー
    print("\n📈 予測値の統計サマリー:")
    print(f"   最小値:     {y_pred.min():,.2f}")
    print(f"   25%分位数:  {np.percentile(y_pred, 25):,.2f}")
    print(f"   中央値:     {np.median(y_pred):,.2f}")
    print(f"   75%分位数:  {np.percentile(y_pred, 75):,.2f}")
    print(f"   最大値:     {y_pred.max():,.2f}")
    print(f"   平均値:     {y_pred.mean():,.2f}")
    print(f"   標準偏差:   {y_pred.std():,.2f}")

    # 誤差の統計サマリー
    errors = y_true - y_pred
    print("\n📉 誤差の統計サマリー:")
    print(f"   最小誤差:   {errors.min():,.2f}")
    print(f"   25%分位数:  {np.percentile(errors, 25):,.2f}")
    print(f"   中央値:     {np.median(errors):,.2f}")
    print(f"   75%分位数:  {np.percentile(errors, 75):,.2f}")
    print(f"   最大誤差:   {errors.max():,.2f}")
    print("="*60)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'median_ae': median_ae,
        'std_error': std_error
    }

def train_model_with_visualization(df_features, target_col='Actual Value'):
    """可視化機能付きのモデル学習"""

    print("\n" + "="*60)
    print("🚀 機械学習パイプライン開始")
    print("="*60)

    # データ可視化
    print("\n📊 Step 1: データ分析と可視化")
    print("-"*40)
    visualize_data_distribution(df_features, target_col)
    visualize_seasonal_patterns(df_features, target_col)

    # 特徴量カラムを選択
    feature_cols = [col for col in df_features.columns
                   if col.endswith('_f') or col in ['year', 'month', 'day', 'day_of_week']]

    # 数値型のみを選択
    numeric_cols = []
    for col in feature_cols:
        if col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
            numeric_cols.append(col)

    feature_cols = numeric_cols

    print(f"\n🔍 === 使用する特徴量詳細 ===")
    print(f"✅ 特徴量数: {len(feature_cols)}個")
    print(f"✅ 特徴量リスト: {feature_cols}")

    # ターゲット変数の処理
    if target_col in df_features.columns:
        df_features[target_col] = pd.to_numeric(df_features[target_col], errors='coerce').fillna(0)
    else:
        print(f"エラー: ターゲット変数 '{target_col}' が見つかりません")
        return None, None, None

    # NaNを削除
    df_clean = df_features[feature_cols + [target_col]].dropna()
    print(f"✅ クリーンデータサイズ: {df_clean.shape}")

    # データを分割
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # 時系列データなのでshuffle=False
    )

    print(f"✅ 学習データ: {X_train.shape}")
    print(f"✅ テストデータ: {X_test.shape}")

    # LightGBMモデルの学習
    print("\n📊 Step 2: モデル学習（LightGBM）")
    print("-"*40)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 100,
        'random_state': 42
    }

    model = lgb.LGBMRegressor(**params)

    # コールバックで学習過程を監視
    eval_set = [(X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        callbacks=[
            lgb.early_stopping(10),
            lgb.log_evaluation(20)
        ]
    )

    print(f"✅ 最適イテレーション数: {model.n_estimators_}")

    # 予測
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 評価
    print("\n📊 Step 3: モデル評価")
    print("-"*40)

    print("\n【学習データの評価】")
    train_metrics = print_evaluation_metrics(y_train.values, y_pred_train, "Training Set")

    print("\n【テストデータの評価】")
    test_metrics = print_evaluation_metrics(y_test.values, y_pred_test, "Test Set")

    # モデル性能の可視化
    print("\n📊 Step 4: 結果の可視化")
    print("-"*40)
    visualize_model_performance(y_test.values, y_pred_test, "LightGBM")

    # 特徴量重要度
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n📊 特徴量重要度 TOP10:")
    print("-"*40)
    for idx, row in importance.head(10).iterrows():
        print(f"  {row['feature']:15s} : {row['importance']:8.1f}")

    visualize_feature_importance(importance)

    return model, importance, test_metrics

def main():
    """メイン処理"""

    print("\n" + "="*70)
    print("🎯 ヤマサ確定注文予測 - 機械学習パイプライン（可視化版）")
    print("="*70)

    # ローカルの特徴量ファイルを読み込む
    features_file = "output_data/features/df_features_yamasa_latest.parquet"

    if not os.path.exists(features_file):
        print(f"❌ エラー: 特徴量ファイルが見つかりません: {features_file}")
        print("   先に create_features_local.py を実行してください")
        return None

    print(f"\n📂 特徴量データ読込中...")
    print(f"   ファイル: {features_file}")

    df_features = pd.read_parquet(features_file)
    print(f"✅ 読込完了: {df_features.shape}")

    # データの基本情報を表示
    print(f"\n📊 データ概要:")
    print("-"*40)
    print(f"  行数: {len(df_features):,}")
    print(f"  カラム数: {len(df_features.columns)}")

    # Material Key毎のデータ数
    if 'Material Key' in df_features.columns:
        n_materials = df_features['Material Key'].nunique()
        print(f"  ユニークなMaterial Key数: {n_materials:,}")

    # 日付範囲
    if 'File Date' in df_features.columns:
        df_features['File Date'] = pd.to_datetime(df_features['File Date'], errors='coerce')
        date_min = df_features['File Date'].min()
        date_max = df_features['File Date'].max()
        print(f"  日付範囲: {date_min.strftime('%Y-%m-%d')} 〜 {date_max.strftime('%Y-%m-%d')}")

    # モデル学習と可視化
    model, importance, metrics = train_model_with_visualization(df_features)

    if model is None:
        print("❌ エラー: モデル学習に失敗しました")
        return None

    # 結果を保存
    output_dir = "output_data/models"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # モデルを保存
    import joblib
    model_file = f"{output_dir}/model_visual_{timestamp}.pkl"
    joblib.dump(model, model_file)
    print(f"\n💾 モデル保存: {model_file}")

    # 特徴量重要度を保存
    importance_file = f"{output_dir}/importance_visual_{timestamp}.csv"
    importance.to_csv(importance_file, index=False)
    print(f"💾 特徴量重要度保存: {importance_file}")

    # メトリクスを保存
    metrics_file = f"{output_dir}/metrics_visual_{timestamp}.json"
    pd.Series(metrics).to_json(metrics_file)
    print(f"💾 評価指標保存: {metrics_file}")

    print("\n" + "="*70)
    print("✅ すべての処理が完了しました！")
    print("="*70)
    print("\n📁 出力ファイル:")
    print(f"  - モデル: {model_file}")
    print(f"  - 重要度: {importance_file}")
    print(f"  - 評価指標: {metrics_file}")
    print(f"  - 可視化画像: output_data/visualizations/*.png")

    return model, importance, metrics

if __name__ == "__main__":
    main()