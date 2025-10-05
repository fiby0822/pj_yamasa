#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""休日ロジックのテスト"""

import pandas as pd
import numpy as np

def test_holiday_logic():
    # テスト用の日付データを生成（2024年12月〜2025年1月）
    dates = pd.date_range('2024-12-25', '2025-01-10')
    df = pd.DataFrame({'file_date': dates})

    print("="*60)
    print("営業日フラグのロジック確認")
    print("="*60)

    # 土日を非営業日とする
    is_weekend = df['file_date'].dt.dayofweek.isin([5, 6])

    # 年末（12/30, 12/31）を非営業日とする
    is_year_end = ((df['file_date'].dt.month == 12) &
                   (df['file_date'].dt.day.isin([30, 31])))

    # 日本の祝日（簡易版）
    holidays = []
    for year in [2024, 2025]:
        holidays.extend([
            pd.Timestamp(f'{year}-01-01'),  # 元日
            pd.Timestamp(f'{year}-01-02'),  # 年始休暇
            pd.Timestamp(f'{year}-01-03'),  # 年始休暇
        ])

    is_holiday = df['file_date'].isin(holidays)

    # 営業日フラグ（土日・祝日・年末以外が1）
    df['is_business_day'] = (~(is_weekend | is_year_end | is_holiday)).astype(int)
    df['is_weekend'] = is_weekend.astype(int)
    df['is_holiday'] = is_holiday.astype(int)
    df['is_year_end'] = is_year_end.astype(int)
    df['day_name'] = df['file_date'].dt.strftime('%a')  # 曜日名

    # 結果を表示
    print("\n日付ごとのフラグ状況:")
    print("─"*60)
    for _, row in df.iterrows():
        date_str = row['file_date'].strftime('%Y-%m-%d')
        day_name = row['day_name']
        business = "営業日" if row['is_business_day'] else "休日"

        # 休日の理由
        reasons = []
        if row['is_weekend']:
            reasons.append("土日")
        if row['is_holiday']:
            reasons.append("祝日")
        if row['is_year_end']:
            reasons.append("年末")

        reason_str = f" ({', '.join(reasons)})" if reasons else ""

        print(f"{date_str} ({day_name}): {business}{reason_str}")

    print("\n" + "="*60)
    print("ロジック説明:")
    print("="*60)
    print("1. 土日（Saturday, Sunday）→ 休日")
    print("2. 12/30, 12/31 → 年末休日")
    print("3. 1/1, 1/2, 1/3 → 年始休日")
    print("4. その他の祝日（2/11, 2/23, 4/29, 5/3-5, 8/11, 11/3, 11/23）→ 休日")
    print("5. 上記以外 → 営業日")
    print("="*60)

if __name__ == "__main__":
    test_holiday_logic()