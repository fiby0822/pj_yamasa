"""
Time series feature generation module with train_end_date support.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import jpholiday
from modules.config.feature_window_config import WINDOW_SIZE_CONFIG


def add_timeseries_features(
    _df: pd.DataFrame,
    window_size_config: Optional[Dict[str, Any]] = None,
    start_year: int = 2021,
    end_year: int = 2025,
    model_type: str = "confirmed_order_demand_yamasa",
    train_end_date: Optional[str] = "2024-12-31"
) -> pd.DataFrame:
    """
    ラグなど時系列関連の特徴量を追加する関数。
    material_key × file_date毎にactual_value(実績値)を持つデータフレームを想定。

    Args:
        _df: 入力データフレーム
        window_size_config: 特徴量のウィンドウサイズ設定（Noneの場合はデフォルト設定を使用）
        start_year: 出力データの開始年（デフォルト: 2021）
        end_year: 出力データの終了年（デフォルト: 2025）
        model_type: モデルタイプ（'unofficial', 'use_actual_value_by_category', 'confirmed_order_demand_yamasa'）
        train_end_date: 学習データの終了日（この日付より後のactual_valueは欠損値として扱う）（デフォルト: "2024-12-31"）

    Returns:
        特徴量が追加されたデータフレーム
    """
    # デフォルト設定を使用
    if window_size_config is None:
        window_size_config = WINDOW_SIZE_CONFIG

    df = _df.copy()

    # file_dateをdatetime型に変換
    df['file_date'] = pd.to_datetime(df['file_date'], format='%Y-%m-%d', errors='coerce')

    # train_end_dateが指定された場合、それより後のactual_valueを一時的にNaNにする
    original_actual_values = None
    if train_end_date is not None:
        train_end_date = pd.to_datetime(train_end_date)
        # 元の値を保存
        original_actual_values = df['actual_value'].copy()
        # train_end_dateより後のデータをNaNにする
        mask = df['file_date'] > train_end_date
        df.loc[mask, 'actual_value'] = np.nan
        print(f"Masking {mask.sum()} records after {train_end_date} as NaN for feature generation")

    # データフレームをmaterial_keyとfile_dateでソート
    df.sort_values(by=['material_key', 'file_date'], inplace=True)

    # 年月の特徴量を追加（_f付きのみ保持）
    df["year_f"] = df['file_date'].dt.year.astype("int16")
    df["month_f"] = df['file_date'].dt.month.astype("int8")

    # 追加の日付関連特徴量
    df["day_of_week_f"] = df['file_date'].dt.dayofweek.astype("int8")  # 0=月曜日

    # material_key毎の特徴量作成
    print("Creating material_key features...")
    _create_material_key_features(df, window_size_config)

    # モデルタイプに応じた特徴量作成
    if model_type == "unofficial":
        print("Creating features for unofficial model...")
        _create_unofficial_features(df, window_size_config)
    elif model_type == "use_actual_value_by_category":
        print("Creating features for category-based model...")
        _create_category_features(df, window_size_config)
    elif model_type == "confirmed_order_demand_yamasa":
        print("Creating features for Yamasa confirmed order model...")
        _create_yamasa_features(df, window_size_config)

    # 共通の追加特徴量を作成
    print("Creating additional common features...")
    _create_additional_features(df)

    print("All features done")

    # 元のactual_valueを復元（必要に応じて）
    if original_actual_values is not None:
        df['actual_value'] = original_actual_values
        print("Restored original actual_values")

    # 指定された年の範囲でフィルタリング
    df.query("(@start_year <= year_f) & (year_f <= @end_year)", inplace=True)

    # material_keyの欠損除去
    df.dropna(subset=["material_key"], inplace=True)

    return df.reset_index(drop=True)


def _create_material_key_features(df: pd.DataFrame, window_size_config: Dict[str, Any]) -> None:
    """material_key毎の特徴量を作成"""
    config = window_size_config.get("material_key", {})

    # ラグ特徴量
    for lag in config.get("lag", []):
        df[f'lag_{lag}_f'] = df.groupby('material_key')['actual_value'].shift(lag)

    # 移動平均（リーク防止のため、その時点の実績値を除外）
    for window in config.get("rolling_mean", []):
        df[f'rolling_mean_{window}_f'] = (
            df.groupby('material_key')['actual_value']
            .shift(1)
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )

    # 移動標準偏差（コメントアウトされているが、必要に応じて有効化）
    for window in config.get("rolling_std", []):
        df[f'rolling_std_{window}_f'] = (
            df.groupby('material_key')['actual_value']
            .shift(1)
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )

    # 変動率（コメントアウトされているが、必要に応じて有効化）
    for rate in config.get("rate_of_change", []):
        shifted_values = df.groupby('material_key')['actual_value'].shift(1)
        if not shifted_values.empty:
            df[f'rate_of_change_{rate}_f'] = shifted_values.pct_change()
        else:
            df[f'rate_of_change_{rate}_f'] = np.nan

    # 累積平均（リーク防止のため、その時点の実績値を除外）
    for window in config.get("cumulative_mean", []):
        df[f'cumulative_mean_{window}_f'] = (
            df.groupby('material_key')['actual_value']
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        )

    print("material_key features done")


def _create_unofficial_features(df: pd.DataFrame, window_size_config: Dict[str, Any]) -> None:
    """内示予測用の特徴量を作成"""
    # variety毎の特徴量
    _create_group_features(df, 'variety', window_size_config.get('variety', {}))

    # mill毎の特徴量
    _create_group_features(df, 'mill', window_size_config.get('mill', {}))

    # orderer毎の特徴量
    _create_group_features(df, 'orderer', window_size_config.get('orderer', {}))


def _create_group_features(df: pd.DataFrame, group_name: str, config: Dict[str, Any]) -> None:
    """グループ毎の特徴量を作成する汎用関数"""
    # ラグ特徴量
    for lag in config.get("lag", []):
        df[f'{group_name}_lag_{lag}_f'] = df.groupby(group_name)['actual_value'].shift(lag)

    # 移動平均
    for window in config.get("rolling_mean", []):
        df[f'{group_name}_rolling_mean_{window}_f'] = (
            df.groupby(group_name)['actual_value']
            .shift(1)
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )

    # 移動標準偏差
    for window in config.get("rolling_std", []):
        df[f'{group_name}_rolling_std_{window}_f'] = (
            df.groupby(group_name)['actual_value']
            .shift(1)
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )

    # 変動率
    for rate in config.get("rate_of_change", []):
        shifted_values = df.groupby(group_name)['actual_value'].shift(1)
        if not shifted_values.empty:
            df[f'{group_name}_rate_of_change_{rate}_f'] = shifted_values.pct_change()
        else:
            df[f'{group_name}_rate_of_change_{rate}_f'] = np.nan

    # 累積平均
    for window in config.get("cumulative_mean", []):
        df[f'{group_name}_cumulative_mean_{window}_f'] = (
            df.groupby(group_name)['actual_value']
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        )


def _create_category_features(df: pd.DataFrame, window_size_config: Dict[str, Any]) -> None:
    """確定注文予測用のカテゴリ特徴量を作成"""

    # 重み付き平均の関数を定義
    def weighted_mean_decreasing(y):
        """直近に近いほど重く評価する場合（単調減少）"""
        if len(y) == 0:
            return np.nan
        weights = np.linspace(1.2, -0.8, len(y))
        if len(weights) > 0:
            weights /= weights[len(weights) // 2] if weights[len(weights) // 2] != 0 else 1
        return np.average(y, weights=weights)

    def weighted_mean_increasing(y):
        """直近に近いほど軽く評価する場合（単調増加）"""
        if len(y) == 0:
            return np.nan
        weights = np.linspace(-0.8, 1.2, len(y))
        if len(weights) > 0:
            weights /= weights[len(weights) // 2] if weights[len(weights) // 2] != 0 else 1
        return np.average(y, weights=weights)

    # 各カテゴリの特徴量を作成
    categories = ['base_code', 'customer_code', 'primary_consumer_code', 'delivery_code', 'place']

    for category in categories:
        if category not in window_size_config:
            continue

        config = window_size_config[category]

        # 累積平均
        for window in config.get("cumulative_mean", []):
            df[f'{category}_cumulative_mean_{window}_f'] = (
                df.groupby(category)['actual_value']
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
            )

            # 重み付き累積平均（直近重視）
            df[f'{category}_weighted_cumulative_mean_high_{window}_f'] = (
                df.groupby(category)['actual_value']
                .transform(lambda x: x.rolling(window=window, min_periods=1)
                          .apply(weighted_mean_decreasing, raw=False).shift(1))
            )

            # 重み付き累積平均（過去重視）
            df[f'{category}_weighted_cumulative_mean_low_{window}_f'] = (
                df.groupby(category)['actual_value']
                .transform(lambda x: x.rolling(window=window, min_periods=1)
                          .apply(weighted_mean_increasing, raw=False).shift(1))
            )

        print(f"{category} features done")


def _create_yamasa_features(df: pd.DataFrame, window_size_config: Dict[str, Any]) -> None:
    """ヤマサ確定注文予測用の特徴量を作成"""

    # product_key毎の特徴量
    if 'product_key' in window_size_config:
        config = window_size_config['product_key']

        for lag in config.get("lag", []):
            df[f'product_key_lag_{lag}_f'] = df.groupby('product_key')['actual_value'].shift(lag)

        for window in config.get("cumulative_mean", []):
            df[f'product_key_cumulative_mean_{window}_f'] = (
                df.groupby('product_key')['actual_value']
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
            )

    # store_code毎の特徴量
    if 'store_code' in window_size_config:
        config = window_size_config['store_code']

        for lag in config.get("lag", []):
            df[f'store_code_lag_{lag}_f'] = df.groupby('store_code')['actual_value'].shift(lag)

        for window in config.get("cumulative_mean", []):
            df[f'store_code_cumulative_mean_{window}_f'] = (
                df.groupby('store_code')['actual_value']
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
            )

    print("Yamasa features done")


def _create_additional_features(df: pd.DataFrame) -> None:
    """追加の共通特徴量を作成"""

    # 2. is_business_day_f: 営業日フラグ
    def is_business_day(date):
        """営業日判定（土日祝日以外）"""
        if pd.isna(date):
            return np.nan
        # 土日判定
        if date.weekday() >= 5:  # 5=土曜, 6=日曜
            return 0
        # 祝日判定
        if jpholiday.is_holiday(date):
            return 0
        return 1

    df['is_business_day_f'] = df['file_date'].apply(is_business_day).astype("int8")

    # 3. dow_month_interaction_f: 曜日×月の交互作用
    df['dow_month_interaction_f'] = (df['day_of_week_f'] * df['month_f']).astype("int16")

    # 4. week_to_date_mean_f: 週初からの累積平均
    # 週の開始日を計算（月曜日始まり）
    df['week_start'] = df['file_date'] - pd.to_timedelta(df['file_date'].dt.dayofweek, unit='d')

    # material_key × week_start でグループ化して累積平均を計算
    df['week_to_date_mean_f'] = (
        df.sort_values(['material_key', 'file_date'])
        .groupby(['material_key', 'week_start'])['actual_value']
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=[0,1], drop=True)
        .astype("float32")
    )

    # 現在のデータをシフト（リーク防止）
    df['week_to_date_mean_f'] = (
        df.groupby(['material_key', 'week_start'])['week_to_date_mean_f']
        .shift(1)
    )

    # week_startカラムを削除
    df.drop('week_start', axis=1, inplace=True)

    # 5. material_dow_mean_f: Material Key×曜日の過去平均
    _create_entity_dow_mean(df, 'material_key', 'material_dow_mean_f')

    # 6. product_key_dow_mean_f: product_key×曜日の過去平均
    if 'product_key' in df.columns:
        _create_entity_dow_mean(df, 'product_key', 'product_key_dow_mean_f')

    # 7. store_code_dow_mean_f: store_code×曜日の過去平均
    if 'store_code' in df.columns:
        _create_entity_dow_mean(df, 'store_code', 'store_code_dow_mean_f')

    # 8. category_lvl1_dow_mean_f: category_lvl1×曜日の過去平均
    if 'category_lvl1' in df.columns:
        _create_entity_dow_mean(df, 'category_lvl1', 'category_lvl1_dow_mean_f')

    print("Additional features done")


def _create_entity_dow_mean(df: pd.DataFrame, entity_col: str, feature_name: str) -> None:
    """エンティティ×曜日の過去平均を計算"""
    # エンティティ×曜日でグループ化して累積平均を計算
    df.sort_values([entity_col, 'file_date'], inplace=True)

    # 累積平均を計算（現在の値を除外）
    df[feature_name] = (
        df.groupby([entity_col, 'day_of_week_f'])['actual_value']
        .transform(lambda x: x.expanding(min_periods=1).mean().shift(1))
        .astype("float32")
    )


# エイリアスを作成（後方互換性のため）
_add_timeseries_features = add_timeseries_features