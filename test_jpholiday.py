#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""jpholiday祝日ロジックのテスト"""

import pandas as pd
import numpy as np
import jpholiday

def test_jpholiday_logic():
    # テスト用の日付データを生成（2024年ゴールデンウィーク、年末年始、その他）
    test_dates = [
        # ゴールデンウィーク周辺
        '2024-04-27', '2024-04-28', '2024-04-29', '2024-04-30',
        '2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04',
        '2024-05-05', '2024-05-06',
        # 年末年始
        '2024-12-28', '2024-12-29', '2024-12-30', '2024-12-31',
        '2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04',
        '2025-01-05', '2025-01-06',
        # 成人の日（移動祝日の例）
        '2025-01-11', '2025-01-12', '2025-01-13',
        # スポーツの日（移動祝日の例）
        '2024-10-12', '2024-10-13', '2024-10-14'
    ]

    df = pd.DataFrame({'file_date': pd.to_datetime(test_dates)})

    print("="*70)
    print("jpholidayを使った営業日フラグのテスト")
    print("="*70)

    # 土日を非営業日とする
    is_weekend = df['file_date'].dt.dayofweek.isin([5, 6])

    # 年末（12/30, 12/31）を非営業日とする
    is_year_end = ((df['file_date'].dt.month == 12) &
                   (df['file_date'].dt.day.isin([30, 31])))

    # jpholidayを使用して祝日判定
    def is_japan_holiday(date):
        """jpholidayを使って祝日判定"""
        # 年始休暇（1/2, 1/3）を追加
        if date.month == 1 and date.day in [2, 3]:
            return True
        # jpholidayで祝日判定
        return jpholiday.is_holiday(date)

    is_holiday = df['file_date'].apply(is_japan_holiday)

    # 営業日フラグ（土日・祝日・年末以外が1）
    df['is_business_day'] = (~(is_weekend | is_year_end | is_holiday)).astype(int)
    df['day_name'] = df['file_date'].dt.strftime('%a')

    # jpholidayから祝日名を取得
    def get_holiday_name(date):
        holiday_name = jpholiday.is_holiday_name(date)
        if holiday_name:
            return holiday_name
        if date.month == 1 and date.day in [2, 3]:
            return "年始休暇"
        return None

    df['holiday_name'] = df['file_date'].apply(get_holiday_name)

    # 結果を表示
    print("\n日付ごとのフラグ状況:")
    print("─"*70)
    for _, row in df.iterrows():
        date_str = row['file_date'].strftime('%Y-%m-%d')
        day_name = row['day_name']
        business = "営業日" if row['is_business_day'] else "休日"

        # 休日の理由
        reasons = []
        if day_name in ['Sat', 'Sun']:
            reasons.append("土日")
        if row['holiday_name']:
            reasons.append(f"祝日:{row['holiday_name']}")
        if row['file_date'].month == 12 and row['file_date'].day in [30, 31]:
            reasons.append("年末")

        reason_str = f" ({', '.join(reasons)})" if reasons else ""

        print(f"{date_str} ({day_name}): {business:4s}{reason_str}")

    # 統計情報
    print("\n" + "="*70)
    print("統計情報:")
    print("="*70)
    print(f"総日数: {len(df)}日")
    print(f"営業日: {df['is_business_day'].sum()}日")
    print(f"休日: {(~df['is_business_day'].astype(bool)).sum()}日")

    # jpholidayで取得できる2025年の祝日一覧
    print("\n" + "="*70)
    print("jpholidayから取得した2025年の祝日一覧:")
    print("="*70)
    holidays_2025 = jpholiday.year_holidays(2025)
    for date, name in holidays_2025:
        print(f"{date.strftime('%Y-%m-%d')} ({date.strftime('%a')}): {name}")

if __name__ == "__main__":
    test_jpholiday_logic()