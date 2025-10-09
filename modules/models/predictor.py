#!/usr/bin/env python3
"""
予測実行モジュール
学習済みモデルを使用した予測とバッチ予測の実行
"""
import gc
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import xgboost as xgb
import boto3
from io import BytesIO
import pickle
import pyarrow.parquet as pq
import pyarrow as pa

warnings.filterwarnings('ignore')

class DemandPredictor:
    """需要予測実行クラス"""

    def __init__(
        self,
        bucket_name: str = "fiby-yamasa-prediction",
        model_type: str = "confirmed_order_demand_yamasa"
    ):
        self.bucket_name = bucket_name
        self.model_type = model_type
        self.s3_client = boto3.client('s3')
        self.models = {}

    def load_models_from_s3(self, period_numbers: Optional[List[int]] = None) -> Dict[int, xgb.XGBRegressor]:
        """
        S3から学習済みモデルをロード

        Parameters:
        -----------
        period_numbers : List[int], optional
            ロードする期間番号のリスト。Noneの場合は最新モデルをロード

        Returns:
        --------
        Dict[int, xgb.XGBRegressor]
            期間番号をキーとするモデル辞書
        """
        if period_numbers is None:
            # 最新モデルを探す
            period_numbers = [1]  # デフォルトは期間1のみ

        loaded_models = {}

        for period_num in period_numbers:
            try:
                # 最新モデルのキー
                model_key = f"output/models/{self.model_type}_model_period{period_num}_latest.pkl"

                # S3からモデルをダウンロード
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=model_key
                )

                # モデルをロード
                model_data = response['Body'].read()
                model = pickle.loads(model_data)
                loaded_models[period_num] = model

                print(f"モデルをロードしました: Period {period_num}")

            except self.s3_client.exceptions.NoSuchKey:
                print(f"Warning: モデルが見つかりません: Period {period_num}")
                continue
            except Exception as e:
                print(f"Error: モデルのロードに失敗しました (Period {period_num}): {e}")
                continue

        self.models = loaded_models
        return loaded_models

    def load_features_from_s3(self, features_path: Optional[str] = None) -> pd.DataFrame:
        """
        S3から特徴量データをロード

        Parameters:
        -----------
        features_path : str, optional
            特徴量ファイルのS3パス。Noneの場合は最新データをロード

        Returns:
        --------
        pd.DataFrame
            特徴量データ
        """
        if features_path is None:
            features_key = f"output/features/{self.model_type}_features_latest.parquet"
        else:
            features_key = features_path

        try:
            # S3からParquetファイルを読み込み
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=features_key
            )
            df = pd.read_parquet(BytesIO(response['Body'].read()))

            print(f"特徴量データをロードしました: {len(df)} レコード")
            return df

        except Exception as e:
            print(f"Error: 特徴量データのロードに失敗しました: {e}")
            raise

    def predict(
        self,
        df: pd.DataFrame,
        model: Optional[xgb.XGBRegressor] = None,
        feature_cols: Optional[List[str]] = None,
        prediction_date: Optional[str] = None,
        batch_size: int = 10000
    ) -> pd.DataFrame:
        """
        予測の実行

        Parameters:
        -----------
        df : pd.DataFrame
            入力データ（特徴量含む）
        model : xgb.XGBRegressor, optional
            使用するモデル。Noneの場合はself.modelsから取得
        feature_cols : List[str], optional
            特徴量列名リスト
        prediction_date : str, optional
            予測対象日
        batch_size : int
            バッチサイズ

        Returns:
        --------
        pd.DataFrame
            予測結果を含むDataFrame
        """
        if model is None:
            if not self.models:
                raise ValueError("モデルがロードされていません")
            model = list(self.models.values())[0]

        # 特徴量列の自動検出
        if feature_cols is None:
            exclude_cols = ['date', 'material_key', 'count_sum_per_dc', 'count', 'target']
            feature_cols = [col for col in df.columns if col not in exclude_cols]

        # 予測対象データのフィルタリング
        if prediction_date is not None:
            df = df[df['date'] == pd.to_datetime(prediction_date)].copy()

        if len(df) == 0:
            print("Warning: 予測対象データがありません")
            return pd.DataFrame()

        # バッチ処理で予測
        predictions = []
        n_batches = (len(df) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            # 特徴量の準備
            X_batch = batch_df[feature_cols].values

            # 予測
            y_pred = model.predict(X_batch)
            y_pred = np.maximum(y_pred, 0)  # 負の値を0に

            predictions.extend(y_pred.tolist())

            if batch_idx % 10 == 0:
                print(f"  バッチ {batch_idx + 1}/{n_batches} 完了")

        # 結果をDataFrameに追加
        result_df = df.copy()
        result_df['predicted'] = predictions

        return result_df

    def predict_future(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
        model: Optional[xgb.XGBRegressor] = None,
        feature_cols: Optional[List[str]] = None,
        save_results: bool = True
    ) -> pd.DataFrame:
        """
        将来期間の予測

        Parameters:
        -----------
        df : pd.DataFrame
            特徴量データ
        start_date : str
            予測開始日
        end_date : str
            予測終了日
        model : xgb.XGBRegressor, optional
            使用するモデル
        feature_cols : List[str], optional
            特徴量列名リスト
        save_results : bool
            結果をS3に保存するか

        Returns:
        --------
        pd.DataFrame
            予測結果
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # 予測期間のデータをフィルタ
        df['date'] = pd.to_datetime(df['date'])
        prediction_df = df[(df['date'] >= start) & (df['date'] <= end)].copy()

        if len(prediction_df) == 0:
            print(f"Warning: {start_date} から {end_date} の期間にデータがありません")
            return pd.DataFrame()

        print(f"予測期間: {start_date} ~ {end_date}")
        print(f"予測対象: {len(prediction_df)} レコード")

        # 予測実行
        result_df = self.predict(
            prediction_df,
            model=model,
            feature_cols=feature_cols
        )

        # 結果の保存
        if save_results:
            self._save_predictions_to_s3(result_df, start_date, end_date)

        return result_df

    def predict_by_material_key(
        self,
        df: pd.DataFrame,
        material_keys: List[str],
        model: Optional[xgb.XGBRegressor] = None,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        特定のMaterial Keyに対する予測

        Parameters:
        -----------
        df : pd.DataFrame
            特徴量データ
        material_keys : List[str]
            対象Material Keyのリスト
        model : xgb.XGBRegressor, optional
            使用するモデル
        feature_cols : List[str], optional
            特徴量列名リスト

        Returns:
        --------
        pd.DataFrame
            予測結果
        """
        # 対象Material Keyのデータをフィルタ
        filtered_df = df[df['material_key'].isin(material_keys)].copy()

        if len(filtered_df) == 0:
            print(f"Warning: 指定されたMaterial Keyのデータがありません")
            return pd.DataFrame()

        print(f"予測対象Material Key数: {len(material_keys)}")
        print(f"予測対象レコード数: {len(filtered_df)}")

        # 予測実行
        result_df = self.predict(
            filtered_df,
            model=model,
            feature_cols=feature_cols
        )

        return result_df

    def batch_predict_with_walk_forward(
        self,
        df: pd.DataFrame,
        train_end: str = "2024-12-31",
        test_interval_days: int = 14,
        n_periods: int = 1,
        feature_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Walk-forward方式でのバッチ予測

        Parameters:
        -----------
        df : pd.DataFrame
            特徴量データ
        train_end : str
            学習データ終了日
        test_interval_days : int
            各テスト期間の日数
        n_periods : int
            予測期間数
        feature_cols : List[str], optional
            特徴量列名リスト

        Returns:
        --------
        Dict[str, Any]
            予測結果と評価指標
        """
        all_predictions = []
        train_end_date = pd.to_datetime(train_end)

        for period in range(1, n_periods + 1):
            # 対応するモデルをロード
            if period not in self.models:
                print(f"Warning: Period {period} のモデルがありません")
                continue

            model = self.models[period]

            # 予測期間の計算
            period_start = train_end_date + timedelta(days=(period - 1) * test_interval_days + 1)
            period_end = period_start + timedelta(days=test_interval_days - 1)

            print(f"\nPeriod {period}: {period_start.strftime('%Y-%m-%d')} ~ {period_end.strftime('%Y-%m-%d')}")

            # 予測実行
            period_df = df[(df['date'] >= period_start) & (df['date'] <= period_end)].copy()

            if len(period_df) > 0:
                predictions = self.predict(
                    period_df,
                    model=model,
                    feature_cols=feature_cols
                )
                predictions['period'] = period
                all_predictions.append(predictions)

        # 全期間の結果を結合
        if all_predictions:
            final_predictions = pd.concat(all_predictions, ignore_index=True)
        else:
            final_predictions = pd.DataFrame()

        return {
            'predictions': final_predictions,
            'n_periods': n_periods,
            'total_records': len(final_predictions)
        }

    def _save_predictions_to_s3(
        self,
        predictions_df: pd.DataFrame,
        start_date: str,
        end_date: str
    ):
        """予測結果をS3に保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Parquet形式で保存
        table = pa.Table.from_pandas(predictions_df)

        # ファイル名の作成
        date_range = f"{start_date.replace('-', '')}_{end_date.replace('-', '')}"
        output_key = f"output/predictions/{self.model_type}_predictions_{date_range}_{timestamp}.parquet"
        latest_key = f"output/predictions/{self.model_type}_predictions_latest.parquet"

        # S3に保存
        buffer = BytesIO()
        pq.write_table(table, buffer)
        buffer.seek(0)

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=output_key,
            Body=buffer.getvalue()
        )

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=latest_key,
            Body=buffer.getvalue()
        )

        print(f"予測結果を保存しました:")
        print(f"  - s3://{self.bucket_name}/{output_key}")

        # CSVでも保存（確認用）
        csv_buffer = BytesIO()
        predictions_df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_key = output_key.replace('.parquet', '.csv')

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=csv_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )


class ModelInference:
    """モデル推論用のユーティリティクラス"""

    @staticmethod
    def prepare_inference_data(
        df: pd.DataFrame,
        target_date: str,
        material_keys: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        推論用データの準備

        Parameters:
        -----------
        df : pd.DataFrame
            入力データ
        target_date : str
            推論対象日
        material_keys : List[str], optional
            対象Material Key

        Returns:
        --------
        pd.DataFrame
            推論用データ
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        target_date = pd.to_datetime(target_date)

        # 日付でフィルタ
        inference_df = df[df['date'] == target_date]

        # Material Keyでフィルタ
        if material_keys is not None:
            inference_df = inference_df[inference_df['material_key'].isin(material_keys)]

        return inference_df

    @staticmethod
    def aggregate_predictions(
        predictions_df: pd.DataFrame,
        group_by: List[str] = ['date'],
        agg_func: str = 'sum'
    ) -> pd.DataFrame:
        """
        予測結果の集約

        Parameters:
        -----------
        predictions_df : pd.DataFrame
            予測結果
        group_by : List[str]
            集約キー
        agg_func : str
            集約関数 ('sum', 'mean', 'median', 'max', 'min')

        Returns:
        --------
        pd.DataFrame
            集約済み予測結果
        """
        agg_funcs = {
            'sum': np.sum,
            'mean': np.mean,
            'median': np.median,
            'max': np.max,
            'min': np.min
        }

        if agg_func not in agg_funcs:
            raise ValueError(f"Invalid aggregation function: {agg_func}")

        # 集約
        aggregated = predictions_df.groupby(group_by).agg({
            'predicted': agg_funcs[agg_func],
            'material_key': 'count'  # カウント
        }).reset_index()

        aggregated.rename(columns={
            'predicted': f'predicted_{agg_func}',
            'material_key': 'n_materials'
        }, inplace=True)

        return aggregated