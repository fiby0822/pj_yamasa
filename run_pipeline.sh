#!/bin/bash
# 全体のパイプライン実行スクリプト

# 仮想環境を有効化
source venv/bin/activate

echo "=================================================="
echo "ヤマサ確定注文予測パイプライン実行"
echo "=================================================="

# オプション引数の処理
CREATE_FEATURES=true
USE_SAVED_FEATURES=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-features)
            CREATE_FEATURES=false
            shift
            ;;
        --no-saved-features)
            USE_SAVED_FEATURES=false
            shift
            ;;
        --features-file)
            FEATURES_FILE="$2"
            USE_SAVED_FEATURES=false
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 1. 特徴量作成（オプション）
if [ "$CREATE_FEATURES" = true ]; then
    echo ""
    echo "Step 1: 特徴量作成中..."
    python create_features.py
else
    echo ""
    echo "Step 1: 特徴量作成をスキップ"
fi

# 2. 学習・予測実行
echo ""
echo "Step 2: 学習・予測実行中..."

if [ "$USE_SAVED_FEATURES" = true ]; then
    # 保存済みの最新特徴量を使用（デフォルト）
    echo "保存済みの最新特徴量ファイルを使用"
    python train_predict.py \
        --is-use-saved-features \
        --train-end-date "2024-12-31" \
        --step-count 6 \
        --use-optuna \
        --n-trials 50
else
    # 指定されたファイルを使用
    if [ -z "$FEATURES_FILE" ]; then
        echo "エラー: --no-saved-features を使用する場合は --features-file を指定してください"
        exit 1
    fi
    echo "指定された特徴量ファイルを使用: $FEATURES_FILE"
    python train_predict.py \
        --no-saved-features \
        --features-file "$FEATURES_FILE" \
        --train-end-date "2024-12-31" \
        --step-count 6 \
        --use-optuna \
        --n-trials 50
fi

echo ""
echo "=================================================="
echo "パイプライン完了"
echo "=================================================="