#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""jpholiday実装前後の精度比較（フォーマット版）"""

import pandas as pd
import boto3
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def create_comparison_table():
    """jpholiday実装前後の比較表を作成"""

    print("=" * 80)
    print("📊 jpholiday実装による予測精度比較")
    print("=" * 80)

    # 比較データ（実際の実行結果から）
    before_jpholiday = {
        "RMSE": 19.15,
        "MAE": 2.59,
        "平均誤差率": 715.97,
        "中央誤差率": 144.00,
        "20%以内_count": 104,
        "20%以内_ratio": 20.8,
        "30%以内_count": 137,
        "30%以内_ratio": 27.4,
        "50%以内_count": 178,
        "50%以内_ratio": 35.6
    }

    after_jpholiday = {
        "RMSE": 19.18,
        "MAE": 2.51,
        "平均誤差率": 618.07,
        "中央誤差率": 121.99,
        "20%以内_count": 120,
        "20%以内_ratio": 24.0,
        "30%以内_count": 152,
        "30%以内_ratio": 30.4,
        "50%以内_count": 183,
        "50%以内_ratio": 36.6
    }

    print("\n### 1. 実装内容\n")
    print("**jpholiday実装前（簡易祝日判定）:**")
    print("- 固定祝日のみ（1/1, 5/3, 5/4, 5/5など）")
    print("- 振替休日なし")
    print("- is_weekend_f, is_holiday_f, is_year_end_fの3つの冗長な特徴量あり")
    print("- 特徴量数: 122個")

    print("\n**jpholiday実装後（正確な祝日判定）:**")
    print("- jpholidayライブラリによる正確な祝日取得")
    print("- 振替休日を自動的に含む")
    print("- 春分の日・秋分の日などの移動祝日も正確")
    print("- 年始休暇（1/2, 1/3）を追加")
    print("- 冗長な特徴量を削除")
    print("- 特徴量数: 119個（-3個）")

    print("\n### 2. 予測精度比較\n")

    # 比較表を作成
    comparison_data = []

    # RMSE
    rmse_diff = after_jpholiday["RMSE"] - before_jpholiday["RMSE"]
    rmse_check = "" if rmse_diff > 0 else " ✅"
    comparison_data.append(["RMSE", f"{before_jpholiday['RMSE']:.2f}",
                           f"{after_jpholiday['RMSE']:.2f}",
                           f"{rmse_diff:+.2f}{rmse_check}"])

    # MAE
    mae_diff = after_jpholiday["MAE"] - before_jpholiday["MAE"]
    mae_check = " ✅" if mae_diff < 0 else ""
    comparison_data.append(["MAE", f"{before_jpholiday['MAE']:.2f}",
                           f"{after_jpholiday['MAE']:.2f}",
                           f"{mae_diff:+.2f}{mae_check}"])

    # 平均誤差率
    mean_err_diff = after_jpholiday["平均誤差率"] - before_jpholiday["平均誤差率"]
    mean_err_check = " ✅" if mean_err_diff < 0 else ""
    comparison_data.append(["平均誤差率", f"{before_jpholiday['平均誤差率']:.2f}%",
                           f"{after_jpholiday['平均誤差率']:.2f}%",
                           f"{mean_err_diff:+.2f}%{mean_err_check}"])

    # 中央誤差率
    median_err_diff = after_jpholiday["中央誤差率"] - before_jpholiday["中央誤差率"]
    median_err_check = " ✅" if median_err_diff < 0 else ""
    comparison_data.append(["中央誤差率", f"{before_jpholiday['中央誤差率']:.2f}%",
                           f"{after_jpholiday['中央誤差率']:.2f}%",
                           f"{median_err_diff:+.2f}%{median_err_check}"])

    # 20%以内
    within20_diff_count = after_jpholiday["20%以内_count"] - before_jpholiday["20%以内_count"]
    within20_diff_ratio = after_jpholiday["20%以内_ratio"] - before_jpholiday["20%以内_ratio"]
    within20_check = " ✅" if within20_diff_count > 0 else ""
    comparison_data.append(["20%以内",
                           f"{before_jpholiday['20%以内_count']}個 ({before_jpholiday['20%以内_ratio']:.1f}%)",
                           f"{after_jpholiday['20%以内_count']}個 ({after_jpholiday['20%以内_ratio']:.1f}%)",
                           f"+{within20_diff_count}個 (+{within20_diff_ratio:.1f}%){within20_check}"])

    # 30%以内
    within30_diff_count = after_jpholiday["30%以内_count"] - before_jpholiday["30%以内_count"]
    within30_diff_ratio = after_jpholiday["30%以内_ratio"] - before_jpholiday["30%以内_ratio"]
    within30_check = " ✅" if within30_diff_count > 0 else ""
    comparison_data.append(["30%以内",
                           f"{before_jpholiday['30%以内_count']}個 ({before_jpholiday['30%以内_ratio']:.1f}%)",
                           f"{after_jpholiday['30%以内_count']}個 ({after_jpholiday['30%以内_ratio']:.1f}%)",
                           f"+{within30_diff_count}個 (+{within30_diff_ratio:.1f}%){within30_check}"])

    # 50%以内
    within50_diff_count = after_jpholiday["50%以内_count"] - before_jpholiday["50%以内_count"]
    within50_diff_ratio = after_jpholiday["50%以内_ratio"] - before_jpholiday["50%以内_ratio"]
    within50_check = " ✅" if within50_diff_count > 0 else ""
    comparison_data.append(["50%以内",
                           f"{before_jpholiday['50%以内_count']}個 ({before_jpholiday['50%以内_ratio']:.1f}%)",
                           f"{after_jpholiday['50%以内_count']}個 ({after_jpholiday['50%以内_ratio']:.1f}%)",
                           f"+{within50_diff_count}個 (+{within50_diff_ratio:.1f}%){within50_check}"])

    # DataFrameとして表示
    df_comparison = pd.DataFrame(comparison_data,
                                 columns=["指標", "jpholiday実装前", "jpholiday実装後", "改善幅"])

    # マークダウン形式のテーブルを生成
    print("| 指標 | jpholiday実装前 | jpholiday実装後 | 改善幅 |")
    print("|------|----------------|----------------|--------|")
    for _, row in df_comparison.iterrows():
        print(f"| {row['指標']} | {row['jpholiday実装前']} | {row['jpholiday実装後']} | {row['改善幅']} |")

    print("\n### 3. 特徴量の変更\n")
    print("**削除された特徴量（冗長）:**")
    print("- is_weekend_f: 土日フラグ → is_business_day_fに含まれる")
    print("- is_holiday_f: 祝日フラグ → is_business_day_fに含まれる")
    print("- is_year_end_f: 年末フラグ → is_business_day_fに含まれる")

    print("\n**改善されたロジック:**")
    print("- is_business_day_f: 土日・祝日・年末（12/30,31）・年始（1/2,3）以外を営業日とする")
    print("- jpholidayによる正確な祝日判定（振替休日含む）")

    print("\n### 4. 主な改善点\n")
    print("- ✅ MAE改善: 2.59 → 2.51 (-0.08)")
    print("- ✅ 予測誤差の大幅改善: 平均97.90%、中央値22.01%削減")
    print("- ✅ 精度向上: 20%以内が3.2%、30%以内が3.0%向上")
    print("- ✅ 特徴量数削減: 122個 → 119個（冗長な特徴量を排除）")
    print("- ⚠️ RMSEは微増（+0.03）だが、全体的な予測精度は改善")

    print("\n### 5. 技術的意義\n")
    print("- **正確な祝日判定**: 振替休日や移動祝日を考慮")
    print("- **特徴量の整理**: 冗長な特徴量を削除し、モデルの解釈性向上")
    print("- **年末年始対応**: 日本の商習慣に合わせた休日設定")
    print("- **予測の安定性**: 中央誤差率の改善により、外れ値の影響を抑制")

    # S3に保存
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

    bucket_name = "fiby-yamasa-prediction"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 比較結果をJSON形式で保存
    import json
    comparison_json = {
        "timestamp": datetime.now().isoformat(),
        "comparison_table": df_comparison.to_dict('records'),
        "before_jpholiday": before_jpholiday,
        "after_jpholiday": after_jpholiday,
        "improvements": {
            "mae_improvement": mae_diff,
            "mean_error_improvement": mean_err_diff,
            "median_error_improvement": median_err_diff,
            "within_20_improvement": within20_diff_count,
            "within_30_improvement": within30_diff_count,
            "within_50_improvement": within50_diff_count,
            "feature_count_reduction": 3
        }
    }

    # S3に保存
    json_content = json.dumps(comparison_json, ensure_ascii=False, indent=2)
    key = f"models/jpholiday_comparison_formatted_{timestamp}.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json_content,
        ContentType='application/json'
    )
    print(f"\n比較結果をS3に保存: s3://{bucket_name}/{key}")

    return df_comparison

if __name__ == "__main__":
    create_comparison_table()