#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""テスト期間の比較結果を保存・表示"""

import boto3
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def save_and_display_comparison():
    # 比較結果データ
    comparison_data = {
        "timestamp": datetime.now().isoformat(),
        "test_periods": {
            "6_months": {
                "period": "2025-01-01 〜 2025-06-30",
                "months": 6,
                "material_keys": 500
            },
            "5_months": {
                "period": "2025-02-01 〜 2025-06-30",
                "months": 5,
                "material_keys": 483
            }
        },
        "model_performance": {
            "6_months": {
                "rmse": 19.15,
                "mae": 2.59
            },
            "5_months": {
                "rmse": 20.42,
                "mae": 2.67
            }
        },
        "error_statistics": {
            "6_months": {
                "mean_error_rate": 715.97,
                "median_error_rate": 144.00,
                "std_error_rate": 1502.02
            },
            "5_months": {
                "mean_error_rate": 463.54,
                "median_error_rate": 82.92,
                "std_error_rate": 1052.37
            }
        },
        "accuracy_distribution": {
            "6_months": {
                "within_20_percent": {"count": 104, "ratio": 20.8},
                "within_30_percent": {"count": 137, "ratio": 27.4},
                "within_50_percent": {"count": 178, "ratio": 35.6}
            },
            "5_months": {
                "within_20_percent": {"count": 103, "ratio": 21.3},
                "within_30_percent": {"count": 142, "ratio": 29.4},
                "within_50_percent": {"count": 196, "ratio": 40.6}
            }
        }
    }

    # 見やすい比較表示
    print("\n" + "="*80)
    print("📊 テスト期間比較結果（6ヶ月 vs 5ヶ月）")
    print("="*80)

    print("\n【テスト期間】")
    print(f"  6ヶ月: 2025/01/01 〜 2025/06/30（{comparison_data['test_periods']['6_months']['material_keys']} Material Keys）")
    print(f"  5ヶ月: 2025/02/01 〜 2025/06/30（{comparison_data['test_periods']['5_months']['material_keys']} Material Keys）")

    print("\n【モデル性能】")
    print("  指標        6ヶ月     5ヶ月     差異")
    print("  ─────────────────────────────────")
    rmse_6 = comparison_data['model_performance']['6_months']['rmse']
    rmse_5 = comparison_data['model_performance']['5_months']['rmse']
    mae_6 = comparison_data['model_performance']['6_months']['mae']
    mae_5 = comparison_data['model_performance']['5_months']['mae']

    print(f"  RMSE:      {rmse_6:7.2f}   {rmse_5:7.2f}   {rmse_5-rmse_6:+.2f}")
    print(f"  MAE:       {mae_6:7.2f}   {mae_5:7.2f}   {mae_5-mae_6:+.2f}")

    print("\n【予測誤差統計】")
    print("  指標              6ヶ月        5ヶ月        差異")
    print("  ──────────────────────────────────────────")
    mean_6 = comparison_data['error_statistics']['6_months']['mean_error_rate']
    mean_5 = comparison_data['error_statistics']['5_months']['mean_error_rate']
    median_6 = comparison_data['error_statistics']['6_months']['median_error_rate']
    median_5 = comparison_data['error_statistics']['5_months']['median_error_rate']
    std_6 = comparison_data['error_statistics']['6_months']['std_error_rate']
    std_5 = comparison_data['error_statistics']['5_months']['std_error_rate']

    print(f"  平均誤差率:    {mean_6:8.2f}%   {mean_5:8.2f}%   {mean_5-mean_6:+.2f}%")
    print(f"  中央誤差率:    {median_6:8.2f}%   {median_5:8.2f}%   {median_5-median_6:+.2f}%")
    print(f"  標準偏差:      {std_6:8.2f}%   {std_5:8.2f}%   {std_5-std_6:+.2f}%")

    print("\n【予測精度分布】")
    print("  範囲           6ヶ月              5ヶ月              差異")
    print("  ──────────────────────────────────────────────────")
    for threshold in ['20', '30', '50']:
        key = f'within_{threshold}_percent'
        count_6 = comparison_data['accuracy_distribution']['6_months'][key]['count']
        ratio_6 = comparison_data['accuracy_distribution']['6_months'][key]['ratio']
        count_5 = comparison_data['accuracy_distribution']['5_months'][key]['count']
        ratio_5 = comparison_data['accuracy_distribution']['5_months'][key]['ratio']

        print(f"  {threshold}%以内:    {count_6:3d}個 ({ratio_6:4.1f}%)    {count_5:3d}個 ({ratio_5:4.1f}%)    {count_5-count_6:+3d}個 ({ratio_5-ratio_6:+.1f}%)")

    print("\n【分析結果】")
    print("  ✅ 5ヶ月テストの方が予測誤差の中央値が大幅に改善（144.00% → 82.92%）")
    print("  ✅ 5ヶ月テストの方が予測誤差の平均値も改善（715.97% → 463.54%）")
    print("  ✅ 50%以内の予測精度も5ヶ月の方が向上（35.6% → 40.6%）")
    print("  ⚠️  RMSEとMAEは6ヶ月の方が若干良好（学習データが多いため）")

    print("\n【結論】")
    print("  5ヶ月テストの方が、1月の異常値の影響を除外できるため、")
    print("  より安定した予測精度評価が可能になっています。")
    print("="*80)

    # S3に保存
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

    bucket_name = "fiby-yamasa-prediction"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    date_dir = datetime.now().strftime('%Y%m%d')

    # JSONとして保存
    json_content = json.dumps(comparison_data, ensure_ascii=False, indent=2)

    # タイムスタンプ付きバージョン
    key = f"models/{date_dir}/test_period_comparison_{timestamp}.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json_content,
        ContentType='application/json'
    )
    print(f"\n比較結果保存: s3://{bucket_name}/{key}")

    # 最新版として保存
    key_latest = "models/test_period_comparison_latest.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=key_latest,
        Body=json_content,
        ContentType='application/json'
    )
    print(f"最新比較結果保存: s3://{bucket_name}/{key_latest}")

if __name__ == "__main__":
    save_and_display_comparison()