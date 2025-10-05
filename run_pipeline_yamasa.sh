#!/bin/bash
# ヤマサ用特徴量生成と予測のパイプライン実行スクリプト

# 仮想環境を有効化
source venv/bin/activate

echo "=================================================="
echo "🎯 ヤマサ確定注文予測パイプライン（完全版）"
echo "=================================================="

# オプション引数の処理
CREATE_FEATURES=true
RUN_PREDICTION=true
RUN_ERROR_ANALYSIS=false
RUN_VISUAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-features)
            CREATE_FEATURES=false
            shift
            ;;
        --skip-prediction)
            RUN_PREDICTION=false
            shift
            ;;
        --with-error-analysis)
            RUN_ERROR_ANALYSIS=true
            shift
            ;;
        --with-visual)
            RUN_VISUAL=true
            shift
            ;;
        --all)
            RUN_ERROR_ANALYSIS=true
            RUN_VISUAL=true
            shift
            ;;
        -h|--help)
            echo "使用方法: $0 [オプション]"
            echo ""
            echo "オプション:"
            echo "  --skip-features        特徴量生成をスキップ"
            echo "  --skip-prediction      予測処理をスキップ"
            echo "  --with-error-analysis  誤差分析も実行"
            echo "  --with-visual          可視化版も実行"
            echo "  --all                  すべての処理を実行"
            echo "  -h, --help             このヘルプを表示"
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            echo "ヘルプを表示するには -h または --help を使用してください"
            exit 1
            ;;
    esac
done

# 開始時刻を記録
START_TIME=$(date +%s)

# 1. 特徴量作成（ヤマサ版）
if [ "$CREATE_FEATURES" = true ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Step 1: ヤマサ用特徴量作成"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "confirmed_order_demand_yamasa分岐での特徴量を生成"
    echo "product_key_lag_*_f, 時間特徴量(_f付き)などを作成"
    echo ""

    python create_features_yamasa.py

    if [ $? -ne 0 ]; then
        echo "❌ エラー: 特徴量作成に失敗しました"
        exit 1
    fi

    echo "✅ 特徴量作成完了"
else
    echo ""
    echo "⏭️  Step 1: 特徴量作成をスキップ"
fi

# 2. 基本予測実行（ローカル版）
if [ "$RUN_PREDICTION" = true ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🤖 Step 2: 学習・予測実行（ローカル版）"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "LightGBMでの学習と予測を実行"
    echo "使用特徴量: _fで終わるカラムのみ"
    echo ""

    python train_predict_local.py

    if [ $? -ne 0 ]; then
        echo "❌ エラー: 予測処理に失敗しました"
        exit 1
    fi

    echo "✅ 予測処理完了"
else
    echo ""
    echo "⏭️  Step 2: 予測処理をスキップ"
fi

# 3. 誤差分析版実行（オプション）
if [ "$RUN_ERROR_ANALYSIS" = true ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📈 Step 3: 予測誤差分析版実行"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "2025/1/1〜2025/6/30のテストデータで評価"
    echo "Material Key毎・Material Key×File Date毎の誤差分析"
    echo ""

    python train_predict_with_error_analysis.py

    if [ $? -ne 0 ]; then
        echo "⚠️  警告: 誤差分析に失敗しました（処理は継続）"
    else
        echo "✅ 誤差分析完了"
    fi
fi

# 4. 可視化版実行（オプション）
if [ "$RUN_VISUAL" = true ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Step 4: 可視化版実行"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "データ分布、季節性、モデル性能を可視化"
    echo ""

    python train_predict_visual.py

    if [ $? -ne 0 ]; then
        echo "⚠️  警告: 可視化処理に失敗しました（処理は継続）"
    else
        echo "✅ 可視化完了"
    fi
fi

# 終了時刻と処理時間を計算
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED_TIME / 60))
ELAPSED_SEC=$((ELAPSED_TIME % 60))

echo ""
echo "=================================================="
echo "✅ パイプライン完了"
echo "=================================================="
echo "処理時間: ${ELAPSED_MIN}分${ELAPSED_SEC}秒"
echo ""
echo "📁 出力ファイル:"
echo "  - 特徴量: output_data/features/df_features_yamasa_latest.parquet"
echo "  - モデル: output_data/models/model_*.pkl"
echo "  - 評価指標: output_data/models/metrics_*.json"

if [ "$RUN_ERROR_ANALYSIS" = true ]; then
    echo "  - 誤差分析: output_data/models/error_analysis_*.csv"
fi

if [ "$RUN_VISUAL" = true ]; then
    echo "  - 可視化: output_data/visualizations/*.png"
fi

echo ""
echo "次のステップ:"
echo "  • 誤差分析を実行: ./run_pipeline_yamasa.sh --with-error-analysis"
echo "  • 可視化を実行: ./run_pipeline_yamasa.sh --with-visual"
echo "  • すべて実行: ./run_pipeline_yamasa.sh --all"