"""
Feature generation with S3 integration.
"""
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from modules.features.timeseries_features import add_timeseries_features
from modules.data_io.s3_handler import S3Handler
from modules.config.feature_window_config import WINDOW_SIZE_CONFIG


class FeatureGeneratorWithS3:
    """S3統合を含む特徴量生成クラス"""

    def __init__(self, bucket_name: str = "fiby-yamasa-prediction"):
        """
        Args:
            bucket_name: S3バケット名
        """
        self.s3_handler = S3Handler(bucket_name)
        self.window_size_config = WINDOW_SIZE_CONFIG

    def generate_and_save_features(
        self,
        df_input: pd.DataFrame,
        output_key: Optional[str] = None,
        model_type: str = "confirmed_order_demand_yamasa",
        start_year: int = 2021,
        end_year: int = 2025,
        train_end_date: Optional[str] = "2024-12-31",
        window_size_config: Optional[Dict[str, Any]] = None,
        save_format: str = "parquet",
        create_latest: bool = True
    ) -> pd.DataFrame:
        """
        特徴量を生成してS3に保存

        Args:
            df_input: 入力データフレーム
            output_key: S3保存先キー（Noneの場合は自動生成）
            model_type: モデルタイプ
            start_year: 出力データの開始年
            end_year: 出力データの終了年
            train_end_date: 学習データの終了日
            window_size_config: ウィンドウサイズ設定（Noneの場合はデフォルト使用）
            save_format: 保存形式（"parquet" または "csv"）
            create_latest: _latestファイルも作成するか

        Returns:
            特徴量が追加されたデータフレーム
        """
        print(f"Generating features for model_type: {model_type}")
        print(f"Period: {start_year}-{end_year}, Train end date: {train_end_date}")

        # 特徴量生成
        df_features = add_timeseries_features(
            df_input,
            window_size_config=window_size_config or self.window_size_config,
            start_year=start_year,
            end_year=end_year,
            model_type=model_type,
            train_end_date=train_end_date
        )

        # output_keyが指定されていない場合は自動生成
        if output_key is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_key = f"output/features/{model_type}_features_{timestamp}.{save_format}"

        # S3に保存（タイムスタンプ付き）
        if save_format == "parquet":
            self.s3_handler.write_parquet(df_features, output_key)
        elif save_format == "csv":
            self.s3_handler.write_csv(df_features, output_key)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")

        print(f"Features saved to S3: s3://{self.s3_handler.bucket_name}/{output_key}")

        # _latestファイルも作成
        if create_latest:
            latest_key = f"output/features/{model_type}_features_latest.{save_format}"
            if save_format == "parquet":
                self.s3_handler.write_parquet(df_features, latest_key)
            elif save_format == "csv":
                self.s3_handler.write_csv(df_features, latest_key)
            print(f"Latest file saved to S3: s3://{self.s3_handler.bucket_name}/{latest_key}")

        print(f"Shape: {df_features.shape}")

        return df_features

    def load_data_from_s3(self, input_key: str, file_type: str = "parquet") -> pd.DataFrame:
        """
        S3からデータを読み込み

        Args:
            input_key: S3のファイルキー
            file_type: ファイル形式（"parquet", "csv", "excel"）

        Returns:
            読み込んだデータフレーム
        """
        if file_type == "parquet":
            return self.s3_handler.read_parquet(input_key)
        elif file_type == "excel":
            return self.s3_handler.read_excel(input_key)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def process_batch_models(
        self,
        df_input: pd.DataFrame,
        start_year: int = 2021,
        end_year: int = 2025,
        train_end_date: Optional[str] = "2024-12-31"
    ) -> Dict[str, pd.DataFrame]:
        """
        複数のモデルタイプの特徴量を一括生成してS3に保存

        Args:
            df_input: 入力データフレーム
            start_year: 出力データの開始年
            end_year: 出力データの終了年
            train_end_date: 学習データの終了日

        Returns:
            モデルタイプをキーとする特徴量DataFrameの辞書
        """
        model_types = [
            "confirmed_order_demand_yamasa",
            "unofficial",
            "use_actual_value_by_category"
        ]

        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_type in model_types:
            print(f"\n{'='*50}")
            print(f"Processing {model_type}")
            print('='*50)

            # 出力キーの生成（output/features配下）
            output_key = f"output/features/{model_type}_features_{timestamp}.parquet"

            # 特徴量生成と保存（_latestも自動生成）
            df_features = self.generate_and_save_features(
                df_input=df_input,
                output_key=output_key,
                model_type=model_type,
                start_year=start_year,
                end_year=end_year,
                train_end_date=train_end_date,
                create_latest=True
            )

            results[model_type] = df_features

        return results


# 使用例のための関数
def example_usage():
    """使用例"""
    # サンプルデータの作成（実際はS3から読み込む）
    import numpy as np

    dates = pd.date_range(start='2021-01-01', end='2025-12-31', freq='MS')
    material_keys = ['MAT001', 'MAT002', 'MAT003']

    data = []
    for material_key in material_keys:
        for date in dates:
            record = {
                'material_key': material_key,
                'file_date': date.strftime('%Y-%m-%d'),
                'actual_value': np.random.randint(100, 1000),
                'product_key': f'PROD{np.random.randint(1, 4):03d}',
                'store_code': f'STORE{np.random.randint(1, 4):03d}',
            }
            data.append(record)

    df_input = pd.DataFrame(data)

    # 特徴量生成器の初期化
    generator = FeatureGeneratorWithS3()

    # 単一モデルの特徴量生成（output_keyを指定しない場合は自動生成）
    df_features = generator.generate_and_save_features(
        df_input=df_input,
        model_type="confirmed_order_demand_yamasa",
        start_year=2021,
        end_year=2025,
        train_end_date="2024-12-31"
    )

    print(f"\nGenerated features shape: {df_features.shape}")
    print(f"Feature columns: {[col for col in df_features.columns if col.endswith('_f')][:10]}")


if __name__ == "__main__":
    example_usage()