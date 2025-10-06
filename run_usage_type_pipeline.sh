#!/bin/bash
# usage_type別モデルパイプライン実行スクリプト

echo "======================================================================"
echo "ヤマサ予測パイプライン（usage_type別モデル版）"
echo "開始時刻: $(date)"
echo "======================================================================"

# Python仮想環境のアクティベート
cd ~/demand_forecast
source venv/bin/activate
cd ~/yamasa

# エラーが発生したら即座に停止
set -e

# ステップ1: データ準備（オプション：--skip-prepareで省略可能）
if [ "$1" != "--skip-prepare" ]; then
    echo ""
    echo "【ステップ1】データ準備（Step1: Excel統合）"
    echo "----------------------------------------------------------------------"
    python3 prepare_yamasa_data_step1.py
    if [ $? -ne 0 ]; then
        echo "エラー: データ準備Step1に失敗しました"
        exit 1
    fi

    echo ""
    echo "【ステップ2】データ準備（Step2: ゼロ補完）"
    echo "----------------------------------------------------------------------"
    python3 prepare_yamasa_data_step2.py
    if [ $? -ne 0 ]; then
        echo "エラー: データ準備Step2に失敗しました"
        exit 1
    fi
else
    echo ""
    echo "【データ準備をスキップ】--skip-prepareオプション指定"
fi

# ステップ3: 特徴量生成（修正版を使用）
echo ""
echo "【ステップ3】特徴量生成（usage_type情報を含む）"
echo "----------------------------------------------------------------------"
python3 create_features_yamasa_fixed.py
if [ $? -ne 0 ]; then
    echo "エラー: 特徴量生成に失敗しました"
    exit 1
fi

# ステップ4: usage_type別モデルで学習・予測
echo ""
echo "【ステップ4】usage_type別モデルで学習・予測"
echo "----------------------------------------------------------------------"
python3 train_predict_with_usage_type.py
if [ $? -ne 0 ]; then
    echo "エラー: モデル学習・予測に失敗しました"
    exit 1
fi

echo ""
echo "======================================================================"
echo "パイプライン完了"
echo "終了時刻: $(date)"
echo ""
echo "【出力ファイル】"
echo "  - 特徴量: S3://fiby-yamasa-prediction/features/df_features_yamasa_latest.parquet"
echo "  - モデル(household): S3://fiby-yamasa-prediction/models/model_household_latest.pkl"
echo "  - モデル(business): S3://fiby-yamasa-prediction/models/model_business_latest.pkl"
echo "  - 予測結果: S3://fiby-yamasa-prediction/predictions/df_predictions_latest.parquet"
echo "  - 特徴量重要度: S3://fiby-yamasa-prediction/importance/feature_importance_latest.parquet"
echo "======================================================================"