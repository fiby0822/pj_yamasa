#!/usr/bin/env python3
"""
予測実行スクリプト
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from modules.models.predictor import DemandPredictor, ModelInference
from modules.evaluation.metrics import ModelEvaluator

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='ヤマサ確定注文需要予測の実行')
    parser.add_argument('--mode', type=str, default='future',
                        choices=['future', 'walk-forward', 'material-key', 'single-date'],
                        help='予測モード')
    parser.add_argument('--start-date', type=str, default=None,
                        help='予測開始日 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='予測終了日 (YYYY-MM-DD)')
    parser.add_argument('--step-count', type=int, default=1,
                        help='予測する月数 (futureモード用、デフォルト: 1)')
    parser.add_argument('--train-end-date', type=str, default='2024-12-31',
                        help='学習データの終了日 (YYYY-MM-DD)')
    parser.add_argument('--target-date', type=str, default=None,
                        help='予測対象日 (single-dateモード用)')
    parser.add_argument('--material-keys', type=str, nargs='+', default=None,
                        help='対象Material Key (material-keyモード用)')
    parser.add_argument('--features-path', type=str, default=None,
                        help='特徴量ファイルのS3パス')
    parser.add_argument('--model-period', type=int, default=1,
                        help='使用するモデルの期間番号')
    parser.add_argument('--aggregate', type=str, default=None,
                        choices=['sum', 'mean', 'median'],
                        help='予測結果の集約方法')
    parser.add_argument('--save-results', action='store_true',
                        help='予測結果をS3に保存')

    args = parser.parse_args()

    print("="*60)
    print(" ヤマサ確定注文需要予測 - 予測実行")
    print("="*60)
    print()
    print(f"予測モード: {args.mode}")

    # 予測器の初期化
    predictor = DemandPredictor(
        bucket_name="fiby-yamasa-prediction",
        model_type="confirmed_order_demand_yamasa"
    )

    # モデルのロード
    print("\nモデルをロード中...")
    if args.mode == 'walk-forward':
        # Walk-forwardの場合は複数期間のモデルをロード
        n_periods = 3  # デフォルト3期間
        models = predictor.load_models_from_s3(list(range(1, n_periods + 1)))
    else:
        # その他のモードでは指定期間のモデルをロード
        models = predictor.load_models_from_s3([args.model_period])

    if not models:
        print("Error: モデルのロードに失敗しました")
        return 1

    # 特徴量データのロード
    print("\n特徴量データをロード中...")
    df = predictor.load_features_from_s3(args.features_path)

    # モードに応じた予測の実行
    if args.mode == 'future':
        # 将来予測モード
        print(f"\n将来予測を実行")
        print(f"  学習データ終了日: {args.train_end_date}")
        print(f"  予測月数: {args.step_count}")

        predictions = predictor.predict_future(
            df=df,
            start_date=args.start_date,
            end_date=args.end_date,
            step_count=args.step_count,
            train_end_date=args.train_end_date,
            model=models[args.model_period],
            save_results=args.save_results
        )

    elif args.mode == 'walk-forward':
        # Walk-forward予測モード
        print("\nWalk-forward予測を実行...")
        results = predictor.batch_predict_with_walk_forward(
            df=df,
            train_end="2024-12-31",
            test_interval_days=14,
            n_periods=len(models)
        )
        predictions = results['predictions']

    elif args.mode == 'material-key':
        # Material Key指定予測モード
        if args.material_keys is None:
            print("Error: --material-keys を指定してください")
            return 1

        print(f"\nMaterial Key指定予測を実行: {len(args.material_keys)} 件")
        predictions = predictor.predict_by_material_key(
            df=df,
            material_keys=args.material_keys,
            model=models[args.model_period]
        )

    elif args.mode == 'single-date':
        # 単一日付予測モード
        if args.target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        else:
            target_date = args.target_date

        print(f"\n単一日付予測を実行: {target_date}")
        inference_df = ModelInference.prepare_inference_data(
            df=df,
            target_date=target_date,
            material_keys=args.material_keys
        )
        predictions = predictor.predict(
            df=inference_df,
            model=models[args.model_period]
        )

    # 予測結果の表示
    if len(predictions) > 0:
        print(f"\n予測完了: {len(predictions)} レコード")

        # 基本統計
        print("\n予測結果の統計:")
        print(f"  平均: {predictions['predicted'].mean():.2f}")
        print(f"  中央値: {predictions['predicted'].median():.2f}")
        print(f"  標準偏差: {predictions['predicted'].std():.2f}")
        print(f"  最小値: {predictions['predicted'].min():.2f}")
        print(f"  最大値: {predictions['predicted'].max():.2f}")

        # 集約が指定されている場合
        if args.aggregate:
            print(f"\n{args.aggregate}での集約結果:")
            aggregated = ModelInference.aggregate_predictions(
                predictions,
                group_by=['date'] if 'date' in predictions.columns else [],
                agg_func=args.aggregate
            )
            print(aggregated.head(10))

        # 結果の保存
        if args.save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_key = f"output/predictions/batch_predictions_{timestamp}.csv"

            import boto3
            from io import StringIO

            s3_client = boto3.client('s3')
            csv_buffer = StringIO()
            predictions.to_csv(csv_buffer, index=False)

            s3_client.put_object(
                Bucket="fiby-yamasa-prediction",
                Key=output_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv'
            )

            print(f"\n予測結果を保存しました:")
            print(f"  s3://fiby-yamasa-prediction/{output_key}")

    else:
        print("\nWarning: 予測結果がありません")

    print("\n予測処理が完了しました！")
    return 0

if __name__ == "__main__":
    exit(main())