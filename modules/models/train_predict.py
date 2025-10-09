#!/usr/bin/env python3
"""
時系列予測モデルの学習・予測モジュール
Walk-forward validationによる時系列クロスバリデーションを実装
"""
import gc
import warnings
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from scipy import stats
import boto3
from io import BytesIO
import pickle
import json

warnings.filterwarnings('ignore')

class TimeSeriesPredictor:
    """時系列予測モデルの学習・予測クラス"""

    def __init__(
        self,
        bucket_name: str = "fiby-yamasa-prediction",
        model_type: str = "confirmed_order_demand_yamasa"
    ):
        self.bucket_name = bucket_name
        self.model_type = model_type
        self.s3_client = boto3.client('s3')

    def train_test_predict_time_split(
        self,
        _df_features: pd.DataFrame,
        train_end_date: str = "2024-12-31",
        step_count: int = 1,
        target_col: str = 'actual_value',
        feature_cols: Optional[List[str]] = None,
        use_optuna: bool = False,
        n_trials: int = 50,
        apply_winsorization: bool = True,
        winsorize_limits: Tuple[float, float] = (0.01, 0.01),
        apply_hampel: bool = True,
        hampel_window: int = 3,
        hampel_threshold: float = 3.0,
        use_gpu: bool = False,
        save_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Any, Dict]:
        """
        Walk-forward validationによる時系列予測（月単位）

        Parameters:
        -----------
        _df_features : pd.DataFrame
            特徴量を含む入力データ
        train_end_date : str
            学習データの終了日
        step_count : int
            予測する月数（各月を1ステップとして予測）
        target_col : str
            予測対象列名
        feature_cols : List[str], optional
            特徴量列名リスト
        use_optuna : bool
            Optunaでハイパーパラメータ最適化を実行するか
        n_trials : int
            Optunaの試行回数
        apply_winsorization : bool
            Winsorization処理を適用するか
        winsorize_limits : Tuple[float, float]
            Winsorization のパーセンタイル
        apply_hampel : bool
            Hampelフィルタを適用するか
        hampel_window : int
            Hampelフィルタのウィンドウサイズ
        hampel_threshold : float
            Hampelフィルタの閾値
        use_gpu : bool
            GPU使用フラグ
        save_dir : str, optional
            モデル保存ディレクトリ（S3パス）
        verbose : bool
            詳細出力フラグ

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Any, Dict]
            - df_pred_all: 全予測結果
            - bykey_df: Material Key別の評価結果
            - imp_last: 最終モデルの特徴量重要度
            - best_params: 最適ハイパーパラメータ
            - model_last: 最終モデル
            - metrics: 評価指標
        """

        # 日付列の準備
        df = _df_features.copy()

        # file_dateをdateとして使用
        if 'file_date' in df.columns and 'date' not in df.columns:
            df['date'] = df['file_date']
        elif 'date' not in df.columns:
            raise ValueError("DataFrame must contain 'date' or 'file_date' column")

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['material_key', 'date'])

        # 特徴量列の自動検出
        if feature_cols is None:
            feature_cols = self._get_feature_columns(df, target_col)

        if verbose:
            print(f"特徴量数: {len(feature_cols)}")
            print(f"対象レコード数: {len(df)}")

        # 学習データの準備
        train_end = pd.to_datetime(train_end_date)

        # 結果格納用
        df_pred_all = []
        bykey_metrics = []
        best_params = None
        model_last = None
        imp_last = None
        all_metrics = {}

        # 月単位でstep_count回予測
        for step in range(1, step_count + 1):
            if verbose:
                print(f"\n===== Step {step}/{step_count} =====")

            # 予測期間の計算（月単位）
            period_test_start = train_end + relativedelta(months=step-1) + timedelta(days=1)
            period_test_end = train_end + relativedelta(months=step)
            period_train_end = train_end + relativedelta(months=step-1)

            if verbose:
                print(f"Train: ~{period_train_end.strftime('%Y-%m-%d')}")
                print(f"Test: {period_test_start.strftime('%Y-%m-%d')} ~ {period_test_end.strftime('%Y-%m-%d')}")

            # データ分割
            train_df = df[df['date'] <= period_train_end].copy()
            test_df = df[(df['date'] >= period_test_start) & (df['date'] <= period_test_end)].copy()

            if len(test_df) == 0:
                if verbose:
                    print(f"Warning: No test data for step {step}")
                continue

            # 外れ値処理（訓練データのみ）
            if apply_winsorization or apply_hampel:
                train_df = self._handle_outliers(
                    train_df, target_col,
                    apply_winsorization, winsorize_limits,
                    apply_hampel, hampel_window, hampel_threshold
                )

            # 訓練データとテストデータの準備
            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_test = test_df[feature_cols].values
            y_test = test_df[target_col].values

            # ハイパーパラメータ最適化（初回のみ）
            if use_optuna and step == 1 and best_params is None:
                if verbose:
                    print("ハイパーパラメータ最適化中...")
                best_params = self._optimize_hyperparameters(
                    X_train, y_train, n_trials, use_gpu
                )
            elif best_params is None:
                best_params = self._get_default_params(use_gpu)

            # モデル学習
            model = xgb.XGBRegressor(
                **best_params,
                random_state=42,
                objective='count:poisson'
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)] if len(X_test) > 0 else None,
                verbose=False
            )

            # 予測（テストデータのみ）
            y_pred = model.predict(X_test)
            y_pred = np.maximum(y_pred, 0)  # 負の予測値を0に

            # 予測結果をDataFrameに追加
            test_df['predicted'] = y_pred
            test_df['step'] = step
            df_pred_all.append(test_df[['date', 'material_key', target_col, 'predicted', 'step']].copy())

            # 最後のモデルと特徴量重要度を保存
            model_last = model
            imp_last = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            # ステップ毎のメトリクスを計算（テストデータのみ）
            step_metrics = self._calculate_metrics(y_test, y_pred)
            all_metrics[f'step_{step}'] = step_metrics

            if verbose:
                print(f"予測完了: {len(y_pred)} サンプル")
                print(f"RMSE: {step_metrics['RMSE']:.4f}, MAE: {step_metrics['MAE']:.4f}")

        # 全予測結果の結合
        if df_pred_all:
            df_pred_all = pd.concat(df_pred_all, ignore_index=True)
            df_pred_all.rename(columns={target_col: 'actual'}, inplace=True)
        else:
            df_pred_all = pd.DataFrame()

        # Material Key別の評価
        bykey_df = pd.DataFrame()
        if len(df_pred_all) > 0:
            for material_key in df_pred_all['material_key'].unique():
                mk_data = df_pred_all[df_pred_all['material_key'] == material_key]
                if len(mk_data) > 0:
                    mk_metrics = self._calculate_metrics(
                        mk_data['actual'].values,
                        mk_data['predicted'].values
                    )
                    mk_metrics['material_key'] = material_key
                    mk_metrics['count'] = len(mk_data)
                    bykey_metrics.append(mk_metrics)

            if bykey_metrics:
                bykey_df = pd.DataFrame(bykey_metrics)

        # 全体メトリクスの計算（全テストデータ）
        if len(df_pred_all) > 0:
            overall_metrics = self._calculate_metrics(
                df_pred_all['actual'].values,
                df_pred_all['predicted'].values
            )
            all_metrics['overall'] = overall_metrics

            if verbose:
                print("\n===== 全体評価指標（テストデータ） =====")
                for metric, value in overall_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"{metric}: {value:.4f}")

        # モデルの保存
        if save_dir and model_last is not None:
            self._save_model_to_s3(model_last, best_params, save_dir)

        return df_pred_all, bykey_df, imp_last, best_params, model_last, all_metrics

    def _get_feature_columns(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """特徴量列の自動検出"""
        exclude_cols = ['date', 'file_date', 'material_key', target_col, 'count', 'target',
                       'product_name', 'file_name', 'step', 'predicted', 'actual']
        feature_cols = []

        for col in df.columns:
            # 除外列はスキップ
            if col in exclude_cols:
                continue

            # 数値型のみを特徴量として使用
            if pd.api.types.is_numeric_dtype(df[col]):
                # NaN率が高い列を除外
                if df[col].isna().mean() < 0.5:
                    feature_cols.append(col)

        return feature_cols

    def _handle_outliers(
        self,
        df: pd.DataFrame,
        target_col: str,
        apply_winsorization: bool,
        winsorize_limits: Tuple[float, float],
        apply_hampel: bool,
        hampel_window: int,
        hampel_threshold: float
    ) -> pd.DataFrame:
        """外れ値処理"""
        df = df.copy()

        if apply_winsorization:
            # Winsorization
            lower = df[target_col].quantile(winsorize_limits[0])
            upper = df[target_col].quantile(1 - winsorize_limits[1])
            df[target_col] = df[target_col].clip(lower, upper)

        if apply_hampel:
            # Hampel filter
            for material_key in df['material_key'].unique():
                mask = df['material_key'] == material_key
                values = df.loc[mask, target_col].values

                # Hampel identifierの実装
                for i in range(len(values)):
                    start = max(0, i - hampel_window)
                    end = min(len(values), i + hampel_window + 1)
                    window = values[start:end]

                    median = np.median(window)
                    mad = np.median(np.abs(window - median))
                    threshold = hampel_threshold * 1.4826 * mad

                    if np.abs(values[i] - median) > threshold:
                        values[i] = median

                df.loc[mask, target_col] = values

        return df

    def _optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int,
        use_gpu: bool
    ) -> Dict[str, Any]:
        """Optunaによるハイパーパラメータ最適化"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            }

            if use_gpu:
                params['tree_method'] = 'gpu_hist'
                params['predictor'] = 'gpu_predictor'

            # Cross validation
            dtrain = xgb.DMatrix(X_train, label=y_train)
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=params['n_estimators'],
                nfold=3,
                metrics='rmse',
                early_stopping_rounds=50,
                verbose_eval=False,
                seed=42,
                obj='count:poisson'
            )

            return cv_results['test-rmse-mean'].min()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        if use_gpu:
            best_params['tree_method'] = 'gpu_hist'
            best_params['predictor'] = 'gpu_predictor'

        return best_params

    def _get_default_params(self, use_gpu: bool) -> Dict[str, Any]:
        """デフォルトパラメータ"""
        params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }

        if use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['predictor'] = 'gpu_predictor'

        return params

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """評価指標の計算"""
        # 基本指標
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

        # エラー率
        error_rates = []
        thresholds = [5, 10, 20, 30, 50]

        for threshold in thresholds:
            error_rate = np.mean(np.abs(y_true - y_pred) > threshold) * 100
            error_rates.append(error_rate)

        # 相関係数
        correlation = np.corrcoef(y_true, y_pred)[0, 1]

        # R2スコア
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Correlation': correlation,
            'R2': r2,
            'Total_Samples': len(y_true),
            'Mean_Actual': np.mean(y_true),
            'Mean_Predicted': np.mean(y_pred),
            'Std_Actual': np.std(y_true),
            'Std_Predicted': np.std(y_pred)
        }

        # エラー率の追加
        for i, threshold in enumerate(thresholds):
            metrics[f'Error_Rate_{threshold}'] = error_rates[i]

        return metrics

    def _save_model_to_s3(self, model: xgb.XGBRegressor, params: Dict, save_dir: Optional[str] = None):
        """モデルをS3に保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # モデルをバイナリ形式で保存
        buffer = BytesIO()
        pickle.dump(model, buffer)
        buffer.seek(0)

        # パラメータも保存
        params_buffer = BytesIO()
        pickle.dump(params, params_buffer)

        if save_dir:
            # カスタムパスへの保存
            model_key = f"{save_dir}/{self.model_type}_model_{timestamp}.pkl"
            params_key = f"{save_dir}/{self.model_type}_params_{timestamp}.pkl"
        else:
            # デフォルトパスへの保存
            model_key = f"output/models/{self.model_type}_model_{timestamp}.pkl"
            params_key = f"output/models/{self.model_type}_params_{timestamp}.pkl"
            latest_model_key = f"output/models/{self.model_type}_model_latest.pkl"
            latest_params_key = f"output/models/{self.model_type}_params_latest.pkl"

            # 最新版も保存
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=latest_model_key,
                Body=buffer.getvalue()
            )

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=latest_params_key,
                Body=params_buffer.getvalue()
            )

        # タイムスタンプ版を保存
        buffer.seek(0)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=model_key,
            Body=buffer.getvalue()
        )

        params_buffer.seek(0)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=params_key,
            Body=params_buffer.getvalue()
        )

        print(f"モデルをS3に保存しました: s3://{self.bucket_name}/{model_key}")