#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""予測品質の分析"""

import pandas as pd
import numpy as np

# 最新の結果を読み込み
mk_improvements = pd.read_csv('material_key_improvements_20251006_230800.csv')

print("="*60)
print("Material Key毎の誤差率分析")
print("="*60)

# 基本統計
print("\n基本モデルの誤差率分布:")
print(f"  総material_key数: {len(mk_improvements)}")
print(f"  平均: {mk_improvements['mean_error_pct_basic'].mean():.1f}%")
print(f"  中央値: {mk_improvements['mean_error_pct_basic'].median():.1f}%")
print(f"  標準偏差: {mk_improvements['mean_error_pct_basic'].std():.1f}%")
print(f"  最小値: {mk_improvements['mean_error_pct_basic'].min():.1f}%")
print(f"  最大値: {mk_improvements['mean_error_pct_basic'].max():.1f}%")

# 分位数
print(f"\n分位数:")
for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    val = mk_improvements['mean_error_pct_basic'].quantile(q)
    print(f"  {q*100:.0f}%点: {val:.1f}%")

# 閾値別のカウント
print(f"\n閾値別のmaterial_key数:")
thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500]
for t in thresholds:
    count = (mk_improvements['mean_error_pct_basic'] <= t).sum()
    pct = count / len(mk_improvements) * 100
    print(f"  {t}%以内: {count}個 ({pct:.1f}%)")

# 100%を超えるものの分析
over_100 = mk_improvements[mk_improvements['mean_error_pct_basic'] > 100]
print(f"\n誤差率100%超のmaterial_key:")
print(f"  個数: {len(over_100)}個 ({len(over_100)/len(mk_improvements)*100:.1f}%)")
if len(over_100) > 0:
    print(f"  平均誤差率: {over_100['mean_error_pct_basic'].mean():.1f}%")
    print(f"  最大誤差率: {over_100['mean_error_pct_basic'].max():.1f}%")

# 20%以内の詳細
within_20 = mk_improvements[mk_improvements['mean_error_pct_basic'] <= 20]
print(f"\n誤差率20%以内のmaterial_key詳細:")
print(f"  個数: {len(within_20)}個")
if len(within_20) > 0:
    print(f"  これらのmaterial_keyの誤差率:")
    for idx, row in within_20.head(10).iterrows():
        print(f"    - {row['material_key']}: {row['mean_error_pct_basic']:.1f}%")

# 改善効果の分析
print(f"\n強化版モデルによる改善:")
print(f"  改善したmaterial_key数: {(mk_improvements['improvement'] > 0).sum()}個")
print(f"  悪化したmaterial_key数: {(mk_improvements['improvement'] < 0).sum()}個")
print(f"  変化なし: {(mk_improvements['improvement'] == 0).sum()}個")

# 強化版での閾値別カウント
print(f"\n強化版モデルの閾値別material_key数:")
for t in [20, 30, 50]:
    count_basic = (mk_improvements['mean_error_pct_basic'] <= t).sum()
    count_enhanced = (mk_improvements['mean_error_pct_enhanced'] <= t).sum()
    diff = count_enhanced - count_basic
    print(f"  {t}%以内: 基本{count_basic}個 → 強化版{count_enhanced}個 (差: {diff:+d}個)")