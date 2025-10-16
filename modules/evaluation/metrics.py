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
        verbose: bool = True,
        train_actual_counts: Dict = None,
        feature_importance: pd.DataFrame = None
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

        # 追加カラムがある場合は追加
        if 'is_over_48_thre' in results:
            pred_df['is_over_48_thre'] = results['is_over_48_thre']
        if 'actual_value_count' in results:
            pred_df['actual_value_count'] = results['actual_value_count']

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
            self._save_evaluation_results(evaluation, train_actual_counts, feature_importance)

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

    def _create_material_summary(self, pred_df: pd.DataFrame, train_actual_counts: Dict = None) -> pd.DataFrame:
        """
        Material Key × Year-Month の集計データを作成

        Parameters:
        -----------
        pred_df : pd.DataFrame
            予測結果のDataFrame
        train_actual_counts : Dict
            学習期間内のMaterial Key別実績発生数

        Returns:
        --------
        pd.DataFrame
            集計結果
        """
        try:
            # DataFrameのコピーを作成
            df = pred_df.copy()

            # dateをdatetime型に変換
            df['date'] = pd.to_datetime(df['date'])

            # Year-Monthカラムを作成
            df['predict_year_month'] = df['date'].dt.strftime('%Y-%m')

            # Material Key × Year-Month でグループ化して集計
            summary = df.groupby(['material_key', 'predict_year_month']).agg({
                'actual': [
                    ('actual_value_count_in_predict_period', lambda x: (x > 0).sum()),  # 予測期間での実績値>0の件数
                    ('actual_value_mean', 'mean')  # 実績値平均
                ],
                'predicted': [
                    ('predict_value_mean', 'mean')  # 予測値平均
                ],
                'error': [
                    ('error_value_mean', 'mean')  # 平均誤差（予測値 - 実績値）
                ]
            }).reset_index()

            # カラム名をフラット化
            summary.columns = [
                'material_key',
                'predict_year_month',
                'actual_value_count_in_predict_period',
                'actual_value_mean',
                'predict_value_mean',
                'error_value_mean'
            ]

            # 学習期間内の実績発生数を追加
            if train_actual_counts:
                summary['actual_value_count_in_train_period'] = summary['material_key'].map(train_actual_counts).fillna(0).astype(int)
            else:
                # train_actual_countsがない場合は0を設定
                summary['actual_value_count_in_train_period'] = 0

            # カラムの順序を調整
            summary = summary[['material_key', 'predict_year_month',
                              'actual_value_count_in_train_period',
                              'actual_value_count_in_predict_period',
                              'actual_value_mean', 'predict_value_mean', 'error_value_mean']]

            # 数値を適切な精度に丸める
            summary['actual_value_mean'] = summary['actual_value_mean'].round(2)
            summary['predict_value_mean'] = summary['predict_value_mean'].round(2)
            summary['error_value_mean'] = summary['error_value_mean'].round(2)

            # ソート
            summary = summary.sort_values(['material_key', 'predict_year_month'])

            return summary

        except Exception as e:
            print(f"Warning: Material集計の作成に失敗しました: {e}")
            return None

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

    def _save_evaluation_results(self, evaluation: Dict[str, Any], train_actual_counts: Dict = None, feature_importance: pd.DataFrame = None):
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

            # Material Key × Year-Month集計CSVの作成と保存
            summary_df = self._create_material_summary(evaluation['prediction_df'], train_actual_counts)
            if summary_df is not None:
                summary_buffer = StringIO()
                summary_df.to_csv(summary_buffer, index=False)
                summary_key = f"output/evaluation/{self.model_type}_material_summary_{timestamp}.csv"
                latest_summary_key = f"output/evaluation/{self.model_type}_material_summary_latest.csv"

                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=summary_key,
                    Body=summary_buffer.getvalue(),
                    ContentType='text/csv'
                )

                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=latest_summary_key,
                    Body=summary_buffer.getvalue(),
                    ContentType='text/csv'
                )

                print(f"  - Material集計: s3://{self.bucket_name}/{summary_key}")

        # 特徴量重要度のCSV保存
        if feature_importance is not None and not feature_importance.empty:
            # 重要度が高い順にソート
            feature_importance_sorted = feature_importance.sort_values('importance', ascending=False)

            importance_buffer = StringIO()
            feature_importance_sorted.to_csv(importance_buffer, index=False)

            # タイムスタンプ付きファイル
            importance_key = f"output/evaluation/{self.model_type}_feature_importance_{timestamp}.csv"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=importance_key,
                Body=importance_buffer.getvalue(),
                ContentType='text/csv'
            )

            # 最新版
            latest_importance_key = f"output/evaluation/{self.model_type}_feature_importance_latest.csv"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=latest_importance_key,
                Body=importance_buffer.getvalue(),
                ContentType='text/csv'
            )

            print(f"  - 特徴量重要度: s3://{self.bucket_name}/{importance_key}")

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

        # 各メトリクスを安全に処理
        rmse = metrics.get('RMSE', 'N/A')
        mae = metrics.get('MAE', 'N/A')
        mape = metrics.get('MAPE', 'N/A')
        r2 = metrics.get('R2', 'N/A')
        corr = metrics.get('Correlation', 'N/A')

        # フォーマット出力（値がN/Aの場合はそのまま出力）
        if rmse != 'N/A':
            print(f"  RMSE: {rmse:.4f}")
        else:
            print(f"  RMSE: {rmse}")

        if mae != 'N/A':
            print(f"  MAE: {mae:.4f}")
        else:
            print(f"  MAE: {mae}")

        if mape != 'N/A':
            print(f"  MAPE: {mape:.2f}%")
        else:
            print(f"  MAPE: {mape}")

        if r2 != 'N/A':
            print(f"  R2 Score: {r2:.4f}")
        else:
            print(f"  R2 Score: {r2}")

        if corr != 'N/A':
            print(f"  相関係数: {corr:.4f}")
        else:
            print(f"  相関係数: {corr}")

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