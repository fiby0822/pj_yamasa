#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage Type specific feature engineering"""

import pandas as pd
import numpy as np

def create_usage_type_features(df):
    """
    Create features specifically for usage_type (business vs household)

    Business characteristics:
    - Higher average values (6x household)
    - Stronger weekday patterns
    - Likely drops more on weekends
    - May have different seasonal patterns

    Household characteristics:
    - Lower average values
    - More stable patterns
    - Less variation between weekdays
    """

    print("Creating usage_type specific features...")

    # 1. Basic encoding
    df['is_business_f'] = (df['usage_type'] == 'business').astype(int)
    df['is_household_f'] = (df['usage_type'] == 'household').astype(int)

    # 2. Usage type × temporal interactions
    df['usage_dow_interaction_f'] = df['is_business_f'] * df['day_of_week_f']
    df['usage_weekend_drop_f'] = df['is_business_f'] * (1 - df['is_business_day_f'])
    df['usage_friday_peak_f'] = df['is_business_f'] * df['is_friday_f']
    df['usage_month_interaction_f'] = df['is_business_f'] * df['month_f']

    # 3. Usage type × weekly milestone interaction (day 7, 14, 21, 28)
    df['usage_milestone_interaction_f'] = df['is_business_f'] * df['is_weekly_milestone_f']

    # Sort for time-series operations
    df = df.sort_values(['material_key', 'usage_type', 'file_date']).reset_index(drop=True)

    # 4. Usage type-specific lag features
    for lag in [1, 3, 7, 14]:
        col_name = f'usage_lag_{lag}_f'
        df[col_name] = df.groupby(['material_key', 'usage_type'])['actual_value'].shift(lag)

    # 5. Usage type-specific rolling statistics
    for window in [3, 7, 14]:
        # Rolling mean
        col_mean = f'usage_rolling_mean_{window}_f'
        df[col_mean] = df.groupby(['material_key', 'usage_type'])['actual_value'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

        # Rolling std
        col_std = f'usage_rolling_std_{window}_f'
        df[col_std] = df.groupby(['material_key', 'usage_type'])['actual_value'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    # 6. Normalized features within usage type
    # Z-score normalization
    df['usage_zscore_f'] = df.groupby(['material_key', 'usage_type'])['actual_value'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    # Min-max scaling
    df['usage_minmax_f'] = df.groupby(['material_key', 'usage_type'])['actual_value'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8) if x.max() > x.min() else 0
    )

    # Percentile within usage type
    df['usage_percentile_f'] = df.groupby('usage_type')['actual_value'].rank(pct=True)

    # 7. Usage type average features
    # Overall usage type mean
    usage_means = df.groupby('usage_type')['actual_value'].mean()
    df['usage_type_mean_f'] = df['usage_type'].map(usage_means)

    # Ratio to usage type mean
    df['usage_ratio_to_mean_f'] = df['actual_value'] / (df['usage_type_mean_f'] + 1e-8)

    # 8. Usage type × day of week specific means
    usage_dow_means = df.groupby(['usage_type', 'day_of_week_f'])['actual_value'].mean()
    df['usage_dow_mean_f'] = df.set_index(['usage_type', 'day_of_week_f']).index.map(usage_dow_means).values

    # Ratio to usage × dow mean
    df['usage_dow_ratio_f'] = df['actual_value'] / (df['usage_dow_mean_f'] + 1e-8)

    # 9. Usage type × month specific patterns
    usage_month_means = df.groupby(['usage_type', 'month_f'])['actual_value'].mean()
    df['usage_month_mean_f'] = df.set_index(['usage_type', 'month_f']).index.map(usage_month_means).values

    # 10. Cross-usage type features (if multiple usage types per material key)
    # Ratio of business to household for same material key
    material_usage_means = df.groupby(['material_key', 'usage_type'])['actual_value'].mean().unstack(fill_value=0)
    if 'business' in material_usage_means.columns and 'household' in material_usage_means.columns:
        material_usage_means['business_household_ratio'] = (
            material_usage_means['business'] / (material_usage_means['household'] + 1e-8)
        )
        df['material_usage_ratio_f'] = df['material_key'].map(
            material_usage_means['business_household_ratio'].to_dict()
        )

    # 11. Usage type-specific cumulative features
    for window in [7, 14, 30]:
        col_name = f'usage_cumulative_mean_{window}_f'
        df[col_name] = df.groupby(['material_key', 'usage_type'])['actual_value'].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )

    # 12. Usage type × product category interaction
    if 'category_lvl1' in df.columns:
        # Encode category as numeric
        category_encoded = df['category_lvl1'].factorize()[0]
        df['usage_category_interaction_f'] = df['is_business_f'] * category_encoded

    # 13. Usage type-specific weekday/weekend difference
    weekday_means = df[df['is_business_day_f'] == 1].groupby(
        ['material_key', 'usage_type']
    )['actual_value'].mean()
    weekend_means = df[df['is_business_day_f'] == 0].groupby(
        ['material_key', 'usage_type']
    )['actual_value'].mean()

    df['usage_weekday_mean_f'] = df.set_index(['material_key', 'usage_type']).index.map(weekday_means).values
    df['usage_weekend_mean_f'] = df.set_index(['material_key', 'usage_type']).index.map(weekend_means).values
    df['usage_weekday_weekend_diff_f'] = df['usage_weekday_mean_f'] - df['usage_weekend_mean_f']

    # Fill NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col.endswith('_f'):
            df[col] = df[col].fillna(0)

    print(f"Created {sum(1 for col in df.columns if col.endswith('_f') and 'usage' in col)} usage_type features")

    return df

def get_usage_type_feature_importance(model, feature_cols):
    """
    Analyze which usage_type features are most important
    """
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    })

    # Filter usage_type related features
    usage_features = importance[importance['feature'].str.contains('usage')]
    usage_features = usage_features.sort_values('importance', ascending=False)

    print("\nTop 10 Usage Type Features:")
    print(usage_features.head(10).to_string(index=False))

    return usage_features

def analyze_usage_type_patterns(df):
    """
    Analyze patterns between business and household usage types
    """
    if 'usage_type' not in df.columns:
        print("No usage_type column found")
        return

    usage_types = df['usage_type'].unique()
    print(f"\nUsage Types Found: {usage_types}")

    if len(usage_types) < 2:
        print("Only one usage type found, cannot compare patterns")
        return

    # Compare statistics
    stats = df.groupby('usage_type')['actual_value'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ])
    print("\nUsage Type Statistics:")
    print(stats)

    # Compare by day of week
    if 'day_of_week_f' in df.columns:
        dow_stats = df.pivot_table(
            values='actual_value',
            index='usage_type',
            columns='day_of_week_f',
            aggfunc='mean'
        )
        print("\nAverage by Day of Week:")
        print(dow_stats.round(2))

        # Calculate weekday vs weekend difference
        df['is_weekend'] = df['day_of_week_f'].isin([5, 6])
        weekend_stats = df.pivot_table(
            values='actual_value',
            index='usage_type',
            columns='is_weekend',
            aggfunc='mean'
        )
        weekend_stats.columns = ['Weekday', 'Weekend']
        print("\nWeekday vs Weekend:")
        print(weekend_stats.round(2))

        # Calculate weekend drop percentage
        weekend_drop = (weekend_stats['Weekday'] - weekend_stats['Weekend']) / weekend_stats['Weekday'] * 100
        print("\nWeekend Drop %:")
        print(weekend_drop.round(2))

    # Compare by month if available
    if 'month_f' in df.columns:
        month_stats = df.pivot_table(
            values='actual_value',
            index='usage_type',
            columns='month_f',
            aggfunc='mean'
        )
        print("\nAverage by Month:")
        print(month_stats.round(2))

    return stats