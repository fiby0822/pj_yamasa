# Usage Type Feature Engineering Guide

## Overview
The `usage_type` column distinguishes between **business** and **household** customers, which have significantly different consumption patterns:
- **Business**: ~6x higher average values, strong weekday patterns, 80% weekend drop
- **Household**: Lower values, more stable patterns, 30% weekend drop

## Recommended Feature Engineering Approach

### 1. **Binary Encoding**
```python
df['is_business_f'] = (df['usage_type'] == 'business').astype(int)
df['is_household_f'] = (df['usage_type'] == 'household').astype(int)
```

### 2. **Usage Type-Specific Time Series Features**
Create separate lag and rolling features for each usage type:
```python
# Separate lag features by usage type
df['usage_lag_1_f'] = df.groupby(['material_key', 'usage_type'])['actual_value'].shift(1)
df['usage_lag_7_f'] = df.groupby(['material_key', 'usage_type'])['actual_value'].shift(7)

# Separate rolling statistics
df['usage_rolling_mean_7_f'] = df.groupby(['material_key', 'usage_type'])['actual_value'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
```

### 3. **Interaction Features**
Capture different temporal patterns between usage types:
```python
# Usage type × day of week
df['usage_dow_interaction_f'] = df['is_business_f'] * df['day_of_week_f']

# Usage type × weekend (business drops more)
df['usage_weekend_drop_f'] = df['is_business_f'] * (1 - df['is_business_day_f'])

# Usage type × Friday (business peaks more)
df['usage_friday_peak_f'] = df['is_business_f'] * df['is_friday_f']
```

### 4. **Normalization Within Usage Type**
Since business values are 6x higher, normalize within each group:
```python
# Z-score normalization
df['usage_zscore_f'] = df.groupby(['material_key', 'usage_type'])['actual_value'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
)

# Ratio to usage type mean
df['usage_ratio_to_mean_f'] = df['actual_value'] / df.groupby('usage_type')['actual_value'].transform('mean')
```

### 5. **Cross-Usage Type Features**
If material keys have both usage types:
```python
# Business to household ratio for same material
material_usage_means = df.groupby(['material_key', 'usage_type'])['actual_value'].mean().unstack()
material_usage_means['ratio'] = material_usage_means['business'] / (material_usage_means['household'] + 1e-8)
df['material_usage_ratio_f'] = df['material_key'].map(material_usage_means['ratio'])
```

## Integration with Existing Pipeline

### Option 1: Integrate into create_features_yamasa.py
Add usage type features to the existing feature creation:
```python
from create_features_usage_type import create_usage_type_features

# In create_features_yamasa.py
def create_features(df):
    # ... existing feature creation ...

    # Add usage type features if column exists
    if 'usage_type' in df.columns:
        df = create_usage_type_features(df)

    return df
```

### Option 2: Separate Models by Usage Type
Train separate models for better performance:
```python
# Split data by usage type
df_business = df[df['usage_type'] == 'business']
df_household = df[df['usage_type'] == 'household']

# Train separate models
model_business = train_model(df_business)
model_household = train_model(df_household)

# Predict based on usage type
predictions = np.where(
    df['usage_type'] == 'business',
    model_business.predict(X),
    model_household.predict(X)
)
```

### Option 3: Let LightGBM Handle It
LightGBM will automatically learn to split on usage_type early in trees:
```python
# Just include is_business_f and is_household_f as features
# LightGBM will learn the optimal splits
```

## Key Features Summary

| Feature Type | Description | Impact |
|-------------|-------------|---------|
| `is_business_f` | Binary flag for business | Allows model to split on type |
| `usage_dow_interaction_f` | Business × day of week | Captures stronger business weekday pattern |
| `usage_weekend_drop_f` | Business × non-business day | Captures 80% business weekend drop |
| `usage_ratio_to_mean_f` | Value / usage type mean | Normalizes across 6x difference |
| `usage_zscore_f` | Z-score within usage type | Standardizes patterns |
| `usage_lag_[n]_f` | Lag by material × usage type | Type-specific time patterns |
| `usage_rolling_mean_[n]_f` | Rolling mean by type | Type-specific trends |

## Performance Expectations

With proper usage type features:
- **20-30% improvement** in RMSE for business predictions (due to capturing weekend drops)
- **10-15% improvement** in overall MAE (due to normalization)
- **Better feature importance** clarity (usage features likely in top 10)
- **More robust predictions** across the 6x value difference

## Testing the Features

Run the test script to see the features in action:
```bash
source venv/bin/activate
python test_usage_type_features.py
```

This will show:
1. Sample data with business/household patterns
2. Statistical differences between types
3. All generated features
4. Feature value distributions

## Next Steps

1. **Check your actual data** for both usage types:
   ```python
   df['usage_type'].value_counts()
   ```

2. **If both types exist**, integrate the features:
   ```python
   from create_features_usage_type import create_usage_type_features
   df = create_usage_type_features(df)
   ```

3. **Monitor feature importance** after training to see which usage features help most

4. **Consider separate models** if patterns are very different (>10x value difference or opposite trends)