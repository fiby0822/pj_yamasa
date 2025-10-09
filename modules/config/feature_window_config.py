"""
Feature window configuration for time series feature generation.
This configuration defines window sizes for various feature types across different data groupings.
"""

WINDOW_SIZE_CONFIG = {
    "material_key": {
        "lag": [1, 2, 3],
        "rolling_mean": [2, 3, 6],
        "rolling_std": [2, 3, 6],
        "rate_of_change": [1],
        # "cumulative_sum": [2, 3, 4, 5, 6, 9, 12],
        "cumulative_mean": [2, 3, 6, 12],
    },
    # 内示予測で使用
    "variety": {
        "lag": [1, 2, 3],
        "rolling_mean": [2, 3, 4, 5, 6],
        "rolling_std": [2, 3, 4, 5, 6],
        "rate_of_change": [1],
        # "cumulative_sum": [2, 3, 4, 5, 6, 9, 12],
        "cumulative_mean": [2, 3, 4, 5, 6, 9, 12],
    },
    "mill": {
        "lag": [1, 2, 3],
        "rolling_mean": [2, 3, 4, 5, 6],
        "rolling_std": [2, 3, 4, 5, 6],  # Fixed typo: was "rbolling_std"
        "rate_of_change": [1],
        # "cumulative_sum": [2, 3, 4, 5, 6, 9, 12],
        "cumulative_mean": [2, 3, 4, 5, 6, 9, 12],
    },
    "orderer": {
        "lag": [1, 2, 3],
        "rolling_mean": [2, 3, 4, 5, 6],
        "rolling_std": [2, 3, 4, 5, 6],
        "rate_of_change": [1],
        # "cumulative_sum": [2, 3, 4, 5, 6, 9, 12],
        "cumulative_mean": [2, 3, 4, 5, 6, 9, 12],
    },
    # 確定注文予測で使用（効いている特徴量がcumulative_meanだけなのでそれのみ使用）
    "base_code": {
        # "lag": [1, 2, 3],
        # "rolling_mean": [2, 3, 4, 5, 6],
        # "rolling_std": [2, 3, 4, 5, 6],
        # "rate_of_change": [1],
        # "cumulative_sum": [2, 3, 4, 5, 6, 9, 12],
        "cumulative_mean": [2, 3, 4, 5, 6, 9, 12],
    },
    "customer_code": {
        # "lag": [1, 2, 3],
        # "rolling_mean": [2, 3, 4, 5, 6],
        # "rolling_std": [2, 3, 4, 5, 6],  # Fixed typo: was "rbolling_std"
        # "rate_of_change": [1],
        # "cumulative_sum": [2, 3, 4, 5, 6, 9, 12],
        "cumulative_mean": [2, 3, 4, 5, 6, 9, 12],
    },
    "primary_consumer_code": {
        # "lag": [1, 2, 3],
        # "rolling_mean": [2, 3, 4, 5, 6],
        # "rolling_std": [2, 3, 4, 5, 6],
        # "rate_of_change": [1],
        # "cumulative_sum": [2, 3, 4, 5, 6, 9, 12],
        "cumulative_mean": [2, 3, 4, 5, 6, 9, 12],
    },
    "delivery_code": {
        # "lag": [1, 2, 3],
        # "rolling_mean": [2, 3, 4, 5, 6],
        # "rolling_std": [2, 3, 4, 5, 6],
        # "rate_of_change": [1],
        # "cumulative_sum": [2, 3, 4, 5, 6],
        "cumulative_mean": [2, 3, 4, 5, 6],
    },
    "place": {
        # "lag": [1, 2, 3],
        # "rolling_mean": [2, 3, 4, 5, 6],
        # "rolling_std": [2, 3, 4, 5, 6],
        # "rate_of_change": [1],
        # "cumulative_sum": [2, 3, 4, 5, 6],
        "cumulative_mean": [2, 3, 4, 5, 6],
    },
    # material_key毎ではなく全体の変数
    "overall": {
        "lag": [1, 2, 3],
        "rolling_mean": [2, 3, 4, 5, 6],
        "rolling_std": [2, 3, 4, 5, 6],
        "rate_of_change": [1],
        # "cumulative_sum": [2, 3, 4, 5, 6],
        "cumulative_mean": [2, 3, 4, 5, 6],
    },
    # 確定注文予測（ヤマサ）で使用
    "product_key": {
        "lag": [1, 2, 3],
        "cumulative_mean": [3, 6, 12],
    },
    "store_code": {
        "lag": [1, 2, 3],
        "cumulative_mean": [3, 6, 12],
    },
    # "category_lvl1": {
    #     "lag": [1, 2, 3],
    #     "cumulative_mean": [3, 6, 12],
    # },
    # "category_lvl2": {
    #     "lag": [1, 2, 3, 4, 5, 6],
    #     "cumulative_mean": [2, 3, 4, 5, 6, 9, 12],
    # },
    # "category_lvl3": {
    #     "lag": [1, 2, 3, 4, 5, 6],
    #     "cumulative_mean": [2, 3, 4, 5, 6, 9, 12],
    # },
}