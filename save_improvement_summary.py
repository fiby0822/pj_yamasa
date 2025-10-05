#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""改善結果サマリーをS3に保存"""

import boto3
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def save_improvement_summary():
    # 改善結果データ
    improvement_data = {
        "timestamp": datetime.now().isoformat(),
        "new_features_added": [
            "is_business_day_f",
            "is_friday_f",
            "dow_month_interaction_f",
            "is_weekly_milestone_f",
            "week_to_date_mean_f",
            "material_dow_mean_f"
        ],
        "total_features": {
            "before": 113,
            "after": 119
        },
        "model_metrics": {
            "rmse": {
                "before": 19.35,
                "after": 19.15
            },
            "mae": {
                "before": 2.64,
                "after": 2.59
            }
        },
        "error_statistics": {
            "mean_error_rate": {
                "before": 965.96,
                "after": 715.97,
                "improvement": 965.96 - 715.97
            },
            "median_error_rate": {
                "before": 224.47,
                "after": 144.00,
                "improvement": 224.47 - 144.00
            }
        },
        "accuracy_distribution": {
            "within_20_percent": {
                "before_count": 89,
                "before_ratio": 17.8,
                "after_count": 104,
                "after_ratio": 20.8,
                "improvement_count": 104 - 89,
                "improvement_ratio": 20.8 - 17.8
            },
            "within_30_percent": {
                "before_count": 115,
                "before_ratio": 23.0,
                "after_count": 137,
                "after_ratio": 27.4,
                "improvement_count": 137 - 115,
                "improvement_ratio": 27.4 - 23.0
            },
            "within_50_percent": {
                "before_count": 150,
                "before_ratio": 30.0,
                "after_count": 178,
                "after_ratio": 35.6,
                "improvement_count": 178 - 150,
                "improvement_ratio": 35.6 - 30.0
            }
        },
        "most_important_feature": "material_dow_mean_f",
        "total_material_keys_analyzed": 500
    }

    # S3クライアント設定
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

    bucket_name = "fiby-yamasa-prediction"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    date_dir = datetime.now().strftime('%Y%m%d')

    # JSONとして保存
    json_content = json.dumps(improvement_data, ensure_ascii=False, indent=2)

    # タイムスタンプ付きバージョン
    key = f"models/{date_dir}/improvement_summary_{timestamp}.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json_content,
        ContentType='application/json'
    )
    print(f"改善サマリー保存: s3://{bucket_name}/{key}")

    # 最新版として保存
    key_latest = "models/improvement_summary_latest.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=key_latest,
        Body=json_content,
        ContentType='application/json'
    )
    print(f"最新改善サマリー保存: s3://{bucket_name}/{key_latest}")

    # 改善結果を見やすく表示
    print("\n" + "="*60)
    print("📊 予測精度改善結果サマリー")
    print("="*60)
    print("\n【追加した特徴量】")
    for feature in improvement_data["new_features_added"]:
        print(f"  - {feature}")

    print("\n【モデル性能の改善】")
    print(f"  RMSE: {improvement_data['model_metrics']['rmse']['before']:.2f} → {improvement_data['model_metrics']['rmse']['after']:.2f}")
    print(f"  MAE: {improvement_data['model_metrics']['mae']['before']:.2f} → {improvement_data['model_metrics']['mae']['after']:.2f}")

    print("\n【予測誤差の改善】")
    print(f"  平均誤差率: {improvement_data['error_statistics']['mean_error_rate']['before']:.2f}% → {improvement_data['error_statistics']['mean_error_rate']['after']:.2f}% (改善: -{improvement_data['error_statistics']['mean_error_rate']['improvement']:.2f}%)")
    print(f"  中央誤差率: {improvement_data['error_statistics']['median_error_rate']['before']:.2f}% → {improvement_data['error_statistics']['median_error_rate']['after']:.2f}% (改善: -{improvement_data['error_statistics']['median_error_rate']['improvement']:.2f}%)")

    print("\n【予測精度分布の改善】")
    for threshold, data in improvement_data["accuracy_distribution"].items():
        threshold_label = threshold.replace("within_", "").replace("_percent", "%以内")
        print(f"  {threshold_label}: {data['before_count']}個 ({data['before_ratio']:.1f}%) → {data['after_count']}個 ({data['after_ratio']:.1f}%) (改善: +{data['improvement_count']}個, +{data['improvement_ratio']:.1f}%)")

    print("\n【最重要特徴量】")
    print(f"  {improvement_data['most_important_feature']}")
    print("="*60)

if __name__ == "__main__":
    save_improvement_summary()