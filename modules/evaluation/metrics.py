#!/usr/bin/env python3
"""
予測モデルの評価指標計算モジュール
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import boto3
from io import BytesIO, StringIO
import json

class ModelEvaluator:
    """モデル評価クラス"""

    def __init__(
        self,
        bucket_name: str = "fiby-yamasa-prediction",
        model_type: str = "confirmed_order_demand_yamasa"
    ):
        self.bucket_name = bucket_name
        self.model_type = model_type
        self.s3_client = boto3.client('s3')

    def evaluate_predictions(
        self,
        results: Dict[str, Any],
        save_results: bool = True,
        generate_plots: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        予測結果の評価

        Parameters:
        -----------
        results : Dict[str, Any]
            train_test_predict_time_split の出力
        save_results : bool
            結果をS3に保存するか
        generate_plots : bool
            可視化を生成するか
        verbose : bool
            詳細出力フラグ

        Returns:
        --------
        Dict[str, Any]
            評価結果
        """

        if len(results.get('predictions', [])) == 0:
            print("Warning: No predictions to evaluate")
            return {}

        # 予測結果のDataFrame作成
        pred_df = pd.DataFrame({
            'date': results['dates'],
            'material_key': results['material_keys'],
            'actual': results['actuals'],
            'predicted': results['predictions']
        })

        pred_df['error'] = pred_df['predicted'] - pred_df['actual']
        pred_df['abs_error'] = np.abs(pred_df['error'])
        pred_df['percentage_error'] = (pred_df['abs_error'] / (pred_df['actual'] + 1e-10)) * 100

        evaluation = {
            'overall_metrics': results.get('metrics', {}),
            'material_key_metrics': {},
            'temporal_metrics': {},
            'error_distribution': {},
            'prediction_df': pred_df
        }

        # Material Key別の評価
        if verbose:
            print("\n===== Material Key別評価 =====")

        material_metrics = []
        for material_key in pred_df['material_key'].unique():
            mk_df = pred_df[pred_df['material_key'] == material_key]
            if len(mk_df) > 0:
                mk_metrics = self._calculate_metrics(
                    mk_df['actual'].values,
                    mk_df['predicted'].values
                )
                mk_metrics['material_key'] = material_key
                mk_metrics['sample_count'] = len(mk_df)
                material_metrics.append(mk_metrics)

        evaluation['material_key_metrics'] = pd.DataFrame(material_metrics)

        if verbose and len(material_metrics) > 0:
            # Top 10 worst performing material keys
            worst_mk = evaluation['material_key_metrics'].nlargest(10, 'RMSE')
            print("\nRMSEが大きいMaterial Key Top 10:")
            for _, row in worst_mk.iterrows():
                print(f"  {row['material_key']}: RMSE={row['RMSE']:.2f}, MAE={row['MAE']:.2f}")

        # 時系列での評価
        if verbose:
            print("\n===== 時系列評価 =====")

        pred_df['date'] = pd.to_datetime(pred_df['date'])
        temporal_metrics = []

        for date in pred_df['date'].unique():
            date_df = pred_df[pred_df['date'] == date]
            if len(date_df) > 0:
                date_metrics = self._calculate_metrics(
                    date_df['actual'].values,
                    date_df['predicted'].values
                )
                date_metrics['date'] = date
                date_metrics['sample_count'] = len(date_df)
                temporal_metrics.append(date_metrics)

        evaluation['temporal_metrics'] = pd.DataFrame(temporal_metrics)

        # エラー分布の分析
        evaluation['error_distribution'] = self._analyze_error_distribution(pred_df)

        if verbose:
            print("\n===== エラー分布 =====")
            for key, value in evaluation['error_distribution'].items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.2f}")

        # 可視化の生成
        if generate_plots:
            plot_paths = self._generate_evaluation_plots(pred_df, evaluation)
            evaluation['plot_paths'] = plot_paths

        # 結果の保存
        if save_results:
            self._save_evaluation_results(evaluation)

        return evaluation

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """評価指標の計算"""
        # ゼロ値を避ける
        epsilon = 1e-10

        # 基本指標
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        # MAPE (ゼロ値対応)
        mape_values = np.abs((y_true - y_pred) / (y_true + epsilon))
        mape = np.mean(mape_values) * 100

        # SMAPE (Symmetric MAPE)
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100

        # 相関係数
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            correlation = 0

        # R2スコア
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + epsilon)) if ss_tot > 0 else 0

        # Median Absolute Error
        median_ae = np.median(np.abs(y_true - y_pred))

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'SMAPE': smape,
            'MedianAE': median_ae,
            'Correlation': correlation,
            'R2': r2
        }

        return metrics

    def _analyze_error_distribution(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
        """エラー分布の分析"""
        errors = pred_df['error'].values
        abs_errors = pred_df['abs_error'].values

        distribution = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'q25_error': np.percentile(errors, 25),
            'q75_error': np.percentile(errors, 75),
            'mean_abs_error': np.mean(abs_errors),
            'median_abs_error': np.median(abs_errors),
            'q95_abs_error': np.percentile(abs_errors, 95),
            'q99_abs_error': np.percentile(abs_errors, 99)
        }

        # エラー閾値別の割合
        thresholds = [5, 10, 20, 30, 50, 100]
        for threshold in thresholds:
            ratio = np.mean(abs_errors <= threshold) * 100
            distribution[f'within_{threshold}_ratio'] = ratio

        # 過大予測と過小予測の割合
        distribution['overestimation_ratio'] = np.mean(errors > 0) * 100
        distribution['underestimation_ratio'] = np.mean(errors < 0) * 100

        return distribution

    def _generate_evaluation_plots(
        self,
        pred_df: pd.DataFrame,
        evaluation: Dict[str, Any]
    ) -> List[str]:
        """評価結果の可視化"""
        plot_paths = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. Actual vs Predicted scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(pred_df['actual'], pred_df['predicted'], alpha=0.5, s=10)
        plt.plot([0, pred_df['actual'].max()], [0, pred_df['actual'].max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        plt.grid(True, alpha=0.3)

        # プロットをS3に保存
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_key = f"output/evaluation/{self.model_type}_actual_vs_predicted_{timestamp}.png"
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=plot_key,
            Body=buffer.getvalue(),
            ContentType='image/png'
        )
        plot_paths.append(plot_key)
        plt.close()

        # 2. Error distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(pred_df['error'], bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', label='Zero error')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_key = f"output/evaluation/{self.model_type}_error_distribution_{timestamp}.png"
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=plot_key,
            Body=buffer.getvalue(),
            ContentType='image/png'
        )
        plot_paths.append(plot_key)
        plt.close()

        # 3. Time series of errors
        if 'temporal_metrics' in evaluation and not evaluation['temporal_metrics'].empty:
            temporal_df = evaluation['temporal_metrics']
            temporal_df['date'] = pd.to_datetime(temporal_df['date'])

            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            plt.plot(temporal_df['date'], temporal_df['RMSE'], marker='o')
            plt.xlabel('Date')
            plt.ylabel('RMSE')
            plt.title('RMSE over Time')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(temporal_df['date'], temporal_df['MAE'], marker='o', color='orange')
            plt.xlabel('Date')
            plt.ylabel('MAE')
            plt.title('MAE over Time')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_key = f"output/evaluation/{self.model_type}_temporal_metrics_{timestamp}.png"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=plot_key,
                Body=buffer.getvalue(),
                ContentType='image/png'
            )
            plot_paths.append(plot_key)
            plt.close()

        # 4. Top worst predictions
        worst_predictions = pred_df.nlargest(20, 'abs_error')
        if len(worst_predictions) > 0:
            plt.figure(figsize=(12, 6))
            x = range(len(worst_predictions))
            width = 0.35

            plt.bar([i - width/2 for i in x], worst_predictions['actual'].values, width, label='Actual', alpha=0.8)
            plt.bar([i + width/2 for i in x], worst_predictions['predicted'].values, width, label='Predicted', alpha=0.8)

            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.title('Top 20 Worst Predictions')
            plt.legend()
            plt.grid(True, alpha=0.3)

            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_key = f"output/evaluation/{self.model_type}_worst_predictions_{timestamp}.png"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=plot_key,
                Body=buffer.getvalue(),
                ContentType='image/png'
            )
            plot_paths.append(plot_key)
            plt.close()

        return plot_paths

    def _save_evaluation_results(self, evaluation: Dict[str, Any]):
        """評価結果をS3に保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 予測結果のCSV保存
        if 'prediction_df' in evaluation:
            csv_buffer = StringIO()
            evaluation['prediction_df'].to_csv(csv_buffer, index=False)
            csv_key = f"output/evaluation/{self.model_type}_predictions_{timestamp}.csv"
            latest_csv_key = f"output/evaluation/{self.model_type}_predictions_latest.csv"

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=csv_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv'
            )

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=latest_csv_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv'
            )

        # メトリクスのJSON保存
        metrics_to_save = {
            'overall_metrics': evaluation.get('overall_metrics', {}),
            'error_distribution': evaluation.get('error_distribution', {})
        }

        # DataFrameを辞書に変換
        if 'material_key_metrics' in evaluation and isinstance(evaluation['material_key_metrics'], pd.DataFrame):
            metrics_to_save['material_key_metrics'] = evaluation['material_key_metrics'].to_dict('records')

        if 'temporal_metrics' in evaluation and isinstance(evaluation['temporal_metrics'], pd.DataFrame):
            temporal_dict = evaluation['temporal_metrics'].copy()
            temporal_dict['date'] = temporal_dict['date'].astype(str)
            metrics_to_save['temporal_metrics'] = temporal_dict.to_dict('records')

        json_key = f"output/evaluation/{self.model_type}_metrics_{timestamp}.json"
        latest_json_key = f"output/evaluation/{self.model_type}_metrics_latest.json"

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=json_key,
            Body=json.dumps(metrics_to_save, indent=2, default=str),
            ContentType='application/json'
        )

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=latest_json_key,
            Body=json.dumps(metrics_to_save, indent=2, default=str),
            ContentType='application/json'
        )

        print(f"評価結果を保存しました:")
        print(f"  - 予測結果: s3://{self.bucket_name}/{csv_key}")
        print(f"  - メトリクス: s3://{self.bucket_name}/{json_key}")


def display_evaluation_summary(evaluation: Dict[str, Any]):
    """評価結果のサマリー表示"""
    print("\n" + "="*60)
    print(" 予測モデル評価サマリー")
    print("="*60)

    # 全体メトリクス
    if 'overall_metrics' in evaluation:
        print("\n【全体評価指標】")
        metrics = evaluation['overall_metrics']
        print(f"  RMSE: {metrics.get('RMSE', 'N/A'):.4f}")
        print(f"  MAE: {metrics.get('MAE', 'N/A'):.4f}")
        print(f"  MAPE: {metrics.get('MAPE', 'N/A'):.2f}%")
        print(f"  R2 Score: {metrics.get('R2', 'N/A'):.4f}")
        print(f"  相関係数: {metrics.get('Correlation', 'N/A'):.4f}")

    # エラー分布
    if 'error_distribution' in evaluation:
        print("\n【エラー分布】")
        dist = evaluation['error_distribution']
        print(f"  平均エラー: {dist.get('mean_error', 'N/A'):.2f}")
        print(f"  標準偏差: {dist.get('std_error', 'N/A'):.2f}")
        print(f"  中央値: {dist.get('median_error', 'N/A'):.2f}")

        print("\n【許容誤差内の割合】")
        for threshold in [5, 10, 20, 30, 50]:
            key = f'within_{threshold}_ratio'
            if key in dist:
                print(f"  ±{threshold}以内: {dist[key]:.1f}%")

    # Material Key別のワースト
    if 'material_key_metrics' in evaluation and not evaluation['material_key_metrics'].empty:
        print("\n【Material Key別ワースト5】")
        worst = evaluation['material_key_metrics'].nlargest(5, 'RMSE')
        for _, row in worst.iterrows():
            print(f"  {row['material_key']}: RMSE={row['RMSE']:.2f}")

    print("\n" + "="*60)