#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test and demonstrate usage_type feature engineering"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data_with_usage_types():
    """
    Create sample data with both business and household usage types
    to demonstrate the different patterns
    """

    # Generate date range
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')

    # Create sample data for 10 material keys, each with both usage types
    data = []

    for material_key in range(1, 11):
        for date in dates:
            dow = date.dayofweek
            month = date.month

            # Business pattern
            # - Higher base value (6x household)
            # - Strong weekday pattern (drops 80% on weekends)
            # - Friday peaks
            # - Monthly seasonality
            business_base = 60 * (1 + 0.1 * np.sin(month * np.pi / 6))  # Monthly variation
            if dow < 5:  # Weekday
                business_value = business_base * (0.8 + 0.05 * dow)  # Increases through week
                if dow == 4:  # Friday peak
                    business_value *= 1.3
            else:  # Weekend
                business_value = business_base * 0.2  # 80% drop

            # Add noise
            business_value *= (1 + np.random.normal(0, 0.1))
            business_value = max(0, business_value)

            # Household pattern
            # - Lower base value
            # - More stable (drops 30% on weekends)
            # - Less day-of-week variation
            household_base = 10 * (1 + 0.05 * np.sin(month * np.pi / 6))
            if dow < 5:  # Weekday
                household_value = household_base * (0.95 + 0.01 * dow)
            else:  # Weekend
                household_value = household_base * 0.7  # 30% drop

            # Add noise
            household_value *= (1 + np.random.normal(0, 0.15))
            household_value = max(0, household_value)

            # Add weekly milestone spikes (day 7, 14, 21, 28)
            if date.day in [7, 14, 21, 28]:
                business_value *= 1.5
                household_value *= 1.2

            # Create records
            data.append({
                'material_key': f'MAT_{material_key:03d}',
                'file_date': date,
                'usage_type': 'business',
                'actual_value': round(business_value, 2),
                'day_of_week_f': dow,
                'month_f': month,
                'is_business_day_f': 1 if dow < 5 else 0,
                'is_friday_f': 1 if dow == 4 else 0,
                'is_weekly_milestone_f': 1 if date.day in [7, 14, 21, 28] else 0
            })

            data.append({
                'material_key': f'MAT_{material_key:03d}',
                'file_date': date,
                'usage_type': 'household',
                'actual_value': round(household_value, 2),
                'day_of_week_f': dow,
                'month_f': month,
                'is_business_day_f': 1 if dow < 5 else 0,
                'is_friday_f': 1 if dow == 4 else 0,
                'is_weekly_milestone_f': 1 if date.day in [7, 14, 21, 28] else 0
            })

    return pd.DataFrame(data)

def demonstrate_usage_type_features():
    """
    Demonstrate the usage_type feature engineering approach
    """

    print("=" * 80)
    print("Usage Type Feature Engineering Demonstration")
    print("=" * 80)

    # Create sample data
    df = create_sample_data_with_usage_types()
    print(f"\nCreated sample data: {df.shape}")
    print(f"Date range: {df['file_date'].min()} to {df['file_date'].max()}")
    print(f"Material keys: {df['material_key'].nunique()}")
    print(f"Usage types: {df['usage_type'].unique()}")

    # Show basic statistics
    print("\n" + "=" * 80)
    print("1. Basic Statistics by Usage Type")
    print("=" * 80)
    stats = df.groupby('usage_type')['actual_value'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ])
    print(stats)

    ratio = stats.loc['business', 'mean'] / stats.loc['household', 'mean']
    print(f"\nMean ratio (business/household): {ratio:.2f}x")

    # Show day of week patterns
    print("\n" + "=" * 80)
    print("2. Day of Week Patterns")
    print("=" * 80)
    dow_stats = df.pivot_table(
        values='actual_value',
        index='usage_type',
        columns='day_of_week_f',
        aggfunc='mean'
    )
    dow_stats.columns = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(dow_stats.round(2))

    # Weekend vs Weekday
    df['is_weekend'] = df['day_of_week_f'].isin([5, 6])
    weekend_stats = df.pivot_table(
        values='actual_value',
        index='usage_type',
        columns='is_weekend',
        aggfunc='mean'
    )
    weekend_stats.columns = ['Weekday', 'Weekend']
    print(f"\nWeekday vs Weekend:")
    print(weekend_stats.round(2))

    weekend_drop = (weekend_stats['Weekday'] - weekend_stats['Weekend']) / weekend_stats['Weekday'] * 100
    print(f"\nWeekend Drop %:")
    print(weekend_drop.round(2))

    # Create usage type features
    print("\n" + "=" * 80)
    print("3. Creating Usage Type Features")
    print("=" * 80)

    # Import the feature creation function
    from create_features_usage_type import create_usage_type_features

    df_with_features = create_usage_type_features(df.copy())

    # List new features
    new_features = [col for col in df_with_features.columns if col.endswith('_f') and col not in df.columns]
    print(f"\nCreated {len(new_features)} new features:")

    # Group features by type
    feature_groups = {
        'Basic Encoding': [f for f in new_features if 'is_business' in f or 'is_household' in f],
        'Interactions': [f for f in new_features if 'interaction' in f],
        'Lag Features': [f for f in new_features if 'lag' in f],
        'Rolling Stats': [f for f in new_features if 'rolling' in f],
        'Normalized': [f for f in new_features if 'zscore' in f or 'minmax' in f or 'percentile' in f],
        'Ratios': [f for f in new_features if 'ratio' in f],
        'Means': [f for f in new_features if 'mean' in f and 'rolling' not in f and 'cumulative' not in f],
        'Cumulative': [f for f in new_features if 'cumulative' in f],
        'Differences': [f for f in new_features if 'diff' in f]
    }

    for group_name, features in feature_groups.items():
        if features:
            print(f"\n{group_name} ({len(features)} features):")
            for f in features[:5]:  # Show first 5
                print(f"  - {f}")
            if len(features) > 5:
                print(f"  ... and {len(features) - 5} more")

    # Show sample of feature values
    print("\n" + "=" * 80)
    print("4. Sample Feature Values")
    print("=" * 80)

    sample_features = [
        'is_business_f',
        'usage_dow_interaction_f',
        'usage_weekend_drop_f',
        'usage_ratio_to_mean_f',
        'usage_zscore_f'
    ]

    sample_data = df_with_features[
        ['material_key', 'file_date', 'usage_type', 'actual_value'] + sample_features
    ].head(20)

    print("\nSample data with features:")
    print(sample_data.to_string(index=False))

    # Compare feature values between usage types
    print("\n" + "=" * 80)
    print("5. Feature Statistics by Usage Type")
    print("=" * 80)

    for feature in sample_features:
        if feature in df_with_features.columns:
            stats = df_with_features.groupby('usage_type')[feature].agg(['mean', 'std', 'min', 'max'])
            print(f"\n{feature}:")
            print(stats.round(3))

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
    Key Insights for Usage Type Feature Engineering:

    1. Business vs Household Patterns:
       - Business has ~6x higher average values
       - Business drops ~80% on weekends vs ~30% for household
       - Business shows stronger day-of-week progression
       - Business has higher Friday peaks

    2. Recommended Features:
       - Separate lag/rolling features by usage type
       - Interaction features (usage × time)
       - Normalized features within usage type
       - Ratio features comparing to usage type means
       - Cross-usage type ratios for same material

    3. Model Considerations:
       - LightGBM will automatically learn to split on usage_type
       - Consider separate models if patterns are very different
       - Use feature importance to identify which features help most
    """)

    return df_with_features

if __name__ == "__main__":
    df_with_features = demonstrate_usage_type_features()