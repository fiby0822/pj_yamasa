#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
学習・予測用スクリプト
S3から特徴量データを読み込み、XGBoostでウォークフォワード予測を実行
"""

import pandas as pd
import numpy as np
import boto3
from io import StringIO, BytesIO
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# XGBoost, scikit-learn, optuna のインポート
try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import optuna
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    print("pip install xgboost scikit-learn optuna を実行してください")
    sys.exit(1)

import matplotlib.pyplot as plt
import seaborn as sns

def train_test_predict_time_split(
    _df_features: pd.DataFrame,
    train_end_date: str,
    step_count: int,
    n_trials: int = 50,
    random_state: int = 42,
    save_dir: str | None = "./model_out",
    run_name: str | None = None,
    save_fig: bool = True,
    use_optuna: bool = True,
    use_winsorize: bool = True,
    winsor_low: float = 0.01,
    winsor_high: float = 0.99,
    use_downweight: bool = True,
    downweight_window: int = 28,
    downweight_nsigma: float = 3.0,
    weight_cap_low: float = 0.25,
    weight_cap_high: float = 1.0,
):
    """
    inplace化したウォークフォワード学習・予測関数
    """
    from pandas.api.types import CategoricalDtype, is_categorical_dtype

    # ---------- utils ----------
    def _safe(s: str) -> str:
        import re as _re
        return _re.sub(r"[^0-9A-Za-z_\-\.]", "-", s)

    def _ensure_dir(d: str | Path):
        if d is None:
            return None
        Path(d).mkdir(parents=True, exist_ok=True)
        return Path(d)

    def month_start(d: pd.Timestamp) -> pd.Timestamp:
        return (d.to_period("M").to_timestamp())

    def month_end(d: pd.Timestamp) -> pd.Timestamp:
        return (d.to_period("M").to_timestamp("M"))

    def next_month_bounds(d: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        nm_start = (d + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()
        nm_end   = (nm_start + pd.offsets.MonthEnd(0))
        return nm_start, nm_end

    # xgboost enable_categorical サポート検知
    _supports_cat = True
    try:
        _ = xgb.XGBRegressor(enable_categorical=True)
    except TypeError:
        _supports_cat = False

    # ---------- clean ----------
    df = _df_features   # copyしない
    df = df.dropna(subset=['actual_value'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how='all')

    df['file_date'] = pd.to_datetime(df['file_date'], errors='coerce')
    df = df.dropna(subset=['file_date'])
    df.sort_values('file_date', inplace=True)

    # 曜日
    if 'day_of_week_mon1' in df.columns and 'day_of_week_mon1_f' not in df.columns:
        df.rename(columns={'day_of_week_mon1': 'day_of_week_mon1_f'}, inplace=True)
    if 'day_of_week_mon1_f' in df.columns:
        dow = pd.to_numeric(df['day_of_week_mon1_f'], errors='coerce').fillna(-1).astype('int16')
        df['day_of_week_mon1_f'] = pd.Categorical(dow)

    # ---------- feature set ----------
    keep_cols = [c for c in df.columns if c.endswith('_f')] + ['actual_value', 'material_key', 'file_date']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_feat = df[keep_cols]   # copyをやめてビュー参照にする

    for c in df_feat.columns:
        if c.endswith('_f') and not is_categorical_dtype(df_feat[c]) and df_feat[c].dtype == 'object':
            df_feat[c] = pd.to_numeric(df_feat[c], errors='coerce')

    drop_all_nan = [c for c in df_feat.columns if c.endswith('_f') and df_feat[c].isna().all()]
    if drop_all_nan:
        df_feat.drop(columns=drop_all_nan, inplace=True)

    # ---------- 外れ値ケア ----------
    def _winsorize_y(train_df: pd.DataFrame, low=0.01, high=0.99) -> pd.Series:
        tmp = train_df[['material_key','file_date','actual_value']].copy()
        tmp['month'] = tmp['file_date'].dt.to_period('M').dt.to_timestamp()

        qs_km = (tmp.loc[tmp['actual_value']>0]
                    .groupby(['material_key','month'])['actual_value']
                    .quantile([low, high]).unstack())
        if qs_km is not None and not qs_km.empty:
            qs_km.columns = ['q_low','q_high']
        tmp = tmp.merge(qs_km, on=['material_key','month'], how='left')

        miss = tmp['q_low'].isna() | tmp['q_high'].isna()
        if miss.any():
            qs_k = (tmp.loc[tmp['actual_value']>0]
                       .groupby('material_key')['actual_value']
                       .quantile([low, high]).unstack())
            if qs_k is not None and not qs_k.empty:
                qs_k.columns = ['q_low_k','q_high_k']
                tmp = tmp.merge(qs_k, on='material_key', how='left')
                tmp['q_low']  = tmp['q_low'].fillna(tmp.get('q_low_k'))
                tmp['q_high'] = tmp['q_high'].fillna(tmp.get('q_high_k'))
                tmp.drop(columns=[c for c in ['q_low_k','q_high_k'] if c in tmp.columns], inplace=True)

        if tmp['q_low'].isna().any() or tmp['q_high'].isna().any():
            pos = tmp.loc[tmp['actual_value']>0, 'actual_value']
            if len(pos):
                ql, qh = pos.quantile(low), pos.quantile(high)
            else:
                ql, qh = tmp['actual_value'].quantile(low), tmp['actual_value'].quantile(high)
            tmp['q_low'].fillna(ql, inplace=True)
            tmp['q_high'].fillna(qh, inplace=True)

        return tmp['actual_value'].clip(lower=tmp['q_low'], upper=tmp['q_high']).astype(float)

    def _hampel_weights(train_df: pd.DataFrame, window=28, nsigma=3.0,
                        cap_low=0.25, cap_high=1.0) -> pd.Series:
        eps = 1e-6
        gdf_sorted = train_df.sort_values(['material_key','file_date'])
        def per_key(gr):
            s = gr['actual_value'].astype(float)
            med = s.rolling(window=window, min_periods=max(1, window//2)).median()
            mad = (s - med).abs().rolling(window=window, min_periods=max(1, window//2)).median()
            sigma = 1.4826 * mad + eps
            z = (s - med).abs() / sigma
            w = np.where(z <= nsigma, 1.0, nsigma / (z + eps))
            w = np.clip(w, cap_low, cap_high)
            return pd.Series(w, index=gr.index)
        w_sorted = gdf_sorted.groupby('material_key', group_keys=False).apply(per_key)
        return w_sorted.reindex(train_df.index).fillna(1.0).astype('float32')

    # ---------- 学習・予測ループ ----------
    cur_train_end = pd.to_datetime(train_end_date)
    all_preds = []
    best_params = None
    best_model_last = None
    importance_df_last = None

    # 固定パラメータ
    fixed_params = {
        'objective': 'count:poisson',
        'max_delta_step': 1,
        'n_estimators': 900,
        'learning_rate': 0.06,
        'max_depth': 6,
        'min_child_weight': 2.0,
        'gamma': 0.0,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'lambda': 1.0,
        'alpha': 0.0,
        'tree_method': 'hist',
        'random_state': random_state,
        'verbosity': 0,
    }
    if _supports_cat:
        fixed_params['enable_categorical'] = True

    out_paths = {}
    base_run_name = run_name or f"xgb_walk_{_safe(train_end_date)}_steps{step_count}" + ("_optuna1st" if use_optuna else "_fixed")

    for step in range(1, step_count + 1):
        test_start, test_end = next_month_bounds(cur_train_end)

        mask_train = df_feat['file_date'] <= cur_train_end
        mask_test  = (df_feat['file_date'] >= test_start) & (df_feat['file_date'] <= test_end)

        if not mask_train.any() or not mask_test.any():
            all_preds.append(pd.DataFrame(columns=['material_key','file_date','actual_value','model_predict_value']))
            cur_train_end = test_end
            continue

        # copyをやめる
        train_df = df_feat.loc[mask_train]
        test_df  = df_feat.loc[mask_test]

        X_train = train_df.drop(columns=['actual_value','material_key','file_date'], errors='ignore')
        y_train = train_df['actual_value'].astype(float)
        X_test  = test_df.drop(columns=['actual_value','material_key','file_date'], errors='ignore')
        y_test  = test_df['actual_value'].astype(float)
        test_keys = test_df[['material_key','file_date']]

        y_train_used = _winsorize_y(train_df, winsor_low, winsor_high) if use_winsorize else y_train
        sample_weight = _hampel_weights(train_df, downweight_window, downweight_nsigma,
                                        weight_cap_low, weight_cap_high).values if use_downweight else None

        # Optunaを使った初回パラメータ探索（step==1のみ）
        if use_optuna and step == 1 and best_params is None:
            print(f"Step {step}: Optuna によるパラメータチューニング開始...")

            def objective(trial):
                params = {
                    'objective': 'count:poisson',
                    'max_delta_step': 1,
                    'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 5),
                    'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'lambda': trial.suggest_float('lambda', 0.5, 3.0),
                    'alpha': trial.suggest_float('alpha', 0.0, 1.0),
                    'tree_method': 'hist',
                    'random_state': random_state,
                    'verbosity': 0,
                }
                if _supports_cat:
                    params['enable_categorical'] = True

                # 内部交差検証（時系列分割）
                tscv = TimeSeriesSplit(n_splits=3)
                cv_scores = []
                for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train_used.iloc[train_idx], y_train_used.iloc[val_idx]
                    w_cv = sample_weight[train_idx] if sample_weight is not None else None

                    model_cv = xgb.XGBRegressor(**params)
                    model_cv.fit(X_cv_train, y_cv_train, sample_weight=w_cv)
                    pred_cv = model_cv.predict(X_cv_val)
                    score = mean_absolute_error(y_cv_val, pred_cv)
                    cv_scores.append(score)

                return np.mean(cv_scores)

            study = optuna.create_study(direction='minimize', study_name='xgboost_walk')
            study.optimize(objective, n_trials=n_trials, n_jobs=1)
            best_params = study.best_params.copy()
            best_params.update({
                'objective': 'count:poisson',
                'max_delta_step': 1,
                'tree_method': 'hist',
                'random_state': random_state,
                'verbosity': 0,
            })
            if _supports_cat:
                best_params['enable_categorical'] = True
            print(f"最適パラメータ: {best_params}")

        # 学習実行
        params_to_use = best_params if best_params else fixed_params
        model = xgb.XGBRegressor(**params_to_use)

        print(f"Step {step}: 学習中... (train: {cur_train_end.date()}, test: {test_start.date()}〜{test_end.date()})")
        model.fit(X_train, y_train_used, sample_weight=sample_weight)

        # 予測
        y_pred = model.predict(X_test)

        # 結果を保存
        test_preds = test_keys.copy()
        test_preds['actual_value'] = y_test.values
        test_preds['model_predict_value'] = y_pred
        all_preds.append(test_preds)

        # 最後のステップの情報を保存
        if step == step_count:
            best_model_last = model
            # 特徴量重要度
            importance = model.feature_importances_
            importance_df_last = pd.DataFrame({
                'feature': X_train.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)

        cur_train_end = test_end

        # ステップごとの評価メトリクス
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"  MAE: {mae:.2f}, RMSE: {np.sqrt(mse):.2f}")

    # ---------- 結果集計 ----------
    df_pred_all = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    # material_key別のエラー率計算
    if not df_pred_all.empty:
        bykey_abs_err_rate_df = df_pred_all.groupby('material_key').apply(
            lambda g: pd.Series({
                'abs_err_rate': (np.abs(g['actual_value'] - g['model_predict_value']).sum() /
                               (g['actual_value'].sum() + 1e-10))
            })
        ).reset_index()
        bykey_abs_err_rate_df = bykey_abs_err_rate_df.sort_values('abs_err_rate')
    else:
        bykey_abs_err_rate_df = pd.DataFrame()

    # 全体メトリクス
    if not df_pred_all.empty:
        overall_mae = mean_absolute_error(df_pred_all['actual_value'], df_pred_all['model_predict_value'])
        overall_mse = mean_squared_error(df_pred_all['actual_value'], df_pred_all['model_predict_value'])
        overall_err_rate = (np.abs(df_pred_all['actual_value'] - df_pred_all['model_predict_value']).sum() /
                          (df_pred_all['actual_value'].sum() + 1e-10))
        metrics = {
            'mae': overall_mae,
            'rmse': np.sqrt(overall_mse),
            'err_rate': overall_err_rate,
            'n_predictions': len(df_pred_all),
            'n_materials': df_pred_all['material_key'].nunique()
        }
    else:
        metrics = {}

    # 保存処理
    if save_dir:
        save_path = _ensure_dir(save_dir)

        # 予測結果
        pred_path = save_path / f"{base_run_name}_predictions.parquet"
        df_pred_all.to_parquet(pred_path, index=False)
        out_paths['predictions'] = str(pred_path)

        # material_key別エラー率
        if not bykey_abs_err_rate_df.empty:
            bykey_path = save_path / f"{base_run_name}_bykey_errors.csv"
            bykey_abs_err_rate_df.to_csv(bykey_path, index=False)
            out_paths['bykey_errors'] = str(bykey_path)

        # 特徴量重要度
        if importance_df_last is not None:
            imp_path = save_path / f"{base_run_name}_importance.csv"
            importance_df_last.to_csv(imp_path, index=False)
            out_paths['importance'] = str(imp_path)

        # パラメータ
        if best_params:
            params_path = save_path / f"{base_run_name}_params.json"
            with open(params_path, 'w') as f:
                json.dump(best_params, f, indent=2, default=str)
            out_paths['params'] = str(params_path)

        # メトリクス
        if metrics:
            metrics_path = save_path / f"{base_run_name}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            out_paths['metrics'] = str(metrics_path)

        print(f"\n保存完了: {save_path}")
        for k, v in out_paths.items():
            print(f"  {k}: {v}")

    return df_pred_all, bykey_abs_err_rate_df, importance_df_last, best_params, best_model_last, metrics

def read_from_s3_parquet(bucket_name, file_key):
    """S3からParquetファイルを読み込む"""
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_parquet(BytesIO(obj['Body'].read()))
        print(f"データ読込成功: s3://{bucket_name}/{file_key}")
        return df
    except Exception as e:
        print(f"S3からの読み込みエラー: {e}")
        raise

def save_to_s3(df, bucket_name, file_key, file_format='parquet'):
    """DataFrameをS3に保存"""
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))
    try:
        if file_format == 'parquet':
            parquet_buffer = BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            s3.put_object(Bucket=bucket_name, Key=file_key, Body=parquet_buffer.getvalue())
        elif file_format == 'csv':
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            s3.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue())
        print(f"データ保存成功: s3://{bucket_name}/{file_key}")
    except Exception as e:
        print(f"S3への保存エラー: {e}")
        raise

def save_json_to_s3(data, bucket_name, file_key):
    """JSONデータをS3に保存"""
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                      region_name=os.getenv('AWS_DEFAULT_REGION'))
    try:
        json_str = json.dumps(data, indent=2, default=str)
        s3.put_object(Bucket=bucket_name, Key=file_key, Body=json_str)
        print(f"JSON保存成功: s3://{bucket_name}/{file_key}")
    except Exception as e:
        print(f"S3への保存エラー: {e}")
        raise

def main():
    """メイン処理"""
    # S3の設定
    bucket_name = "fiby-yamasa-prediction"

    # 特徴量ファイルを指定（create_features.pyの出力を使用）
    # ここは実行時に適切なファイル名を指定してください
    import argparse
    parser = argparse.ArgumentParser(description='学習・予測処理')
    parser.add_argument('--features-file', type=str, default=None,
                       help='特徴量ファイルのS3キー（例: features/df_features_yamasa_20241201_120000.parquet）')
    parser.add_argument('--is-use-saved-features', action='store_true', default=True,
                       help='保存済みの最新特徴量ファイルを使用する（デフォルト: True）')
    parser.add_argument('--no-saved-features', action='store_false', dest='is_use_saved_features',
                       help='保存済み特徴量を使用しない（--features-fileを指定する必要あり）')
    parser.add_argument('--train-end-date', type=str, default='2024-12-31',
                       help='学習データの終了日（例: 2024-12-31）')
    parser.add_argument('--step-count', type=int, default=6,
                       help='予測ステップ数（何ヶ月先まで予測するか）')
    parser.add_argument('--use-optuna', action='store_true', default=True,
                       help='Optunaを使用してパラメータチューニング')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Optunaの試行回数')

    args = parser.parse_args()

    print("=" * 50)
    print("学習・予測処理開始")
    print("=" * 50)

    # 特徴量ファイルのパスを決定
    if args.is_use_saved_features:
        # 保存済みの最新特徴量を使用
        features_file_key = "output_data/features/df_features_yamasa_latest.parquet"
        print(f"\n保存済みの最新特徴量ファイルを使用します: {features_file_key}")

        # メタデータも読込（オプション）
        try:
            metadata_key = "output_data/features/metadata_latest.json"
            s3 = boto3.client('s3',
                            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                            region_name=os.getenv('AWS_DEFAULT_REGION'))
            obj = s3.get_object(Bucket=bucket_name, Key=metadata_key)
            metadata = json.loads(obj['Body'].read().decode('utf-8'))
            print(f"特徴量作成情報:")
            print(f"  - 作成日時: {metadata.get('created_at', 'N/A')}")
            print(f"  - 特徴量数: {metadata.get('n_features', 'N/A')}")
            print(f"  - レコード数: {metadata.get('n_records', 'N/A')}")
            print(f"  - material数: {metadata.get('n_materials', 'N/A')}")
        except Exception as e:
            print(f"メタデータ読込エラー（処理は続行）: {e}")
    else:
        # 指定されたファイルを使用
        if not args.features_file:
            print("エラー: --no-saved-features を使用する場合は --features-file を指定してください")
            sys.exit(1)
        features_file_key = args.features_file
        print(f"\n指定された特徴量ファイルを使用します: {features_file_key}")

    # S3から特徴量データ読込
    print(f"\n1. 特徴量データ読込中...")
    df_features = read_from_s3_parquet(bucket_name, features_file_key)
    print(f"読込データサイズ: {df_features.shape}")

    # 学習・予測実行
    print(f"\n2. ウォークフォワード学習・予測実行中...")
    print(f"学習期間: 〜{args.train_end_date}")
    print(f"予測ステップ数: {args.step_count}ヶ月")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    df_pred_all, bykey_df, imp_last, best_params, model_last, metrics = train_test_predict_time_split(
        _df_features=df_features,
        train_end_date=args.train_end_date,
        step_count=args.step_count,
        use_optuna=args.use_optuna,
        n_trials=args.n_trials,
        save_dir=None,  # S3に直接保存するため、ローカル保存はしない
    )

    # S3に結果を保存
    print(f"\n3. 結果をS3に保存中...")

    # 予測結果
    if not df_pred_all.empty:
        pred_key = f"output_data/predictions/predictions_{timestamp}.parquet"
        save_to_s3(df_pred_all, bucket_name, pred_key, 'parquet')

    # material_key別エラー率
    if not bykey_df.empty:
        bykey_key = f"output_data/predictions/bykey_errors_{timestamp}.csv"
        save_to_s3(bykey_df, bucket_name, bykey_key, 'csv')

    # 特徴量重要度
    if imp_last is not None:
        imp_key = f"output_data/predictions/feature_importance_{timestamp}.csv"
        save_to_s3(imp_last, bucket_name, imp_key, 'csv')
        print("\nTop 20 重要特徴量:")
        print(imp_last.head(20))

    # パラメータ
    if best_params:
        params_key = f"output_data/predictions/params_{timestamp}.json"
        save_json_to_s3(best_params, bucket_name, params_key)

    # メトリクス
    if metrics:
        metrics_key = f"output_data/predictions/metrics_{timestamp}.json"
        save_json_to_s3(metrics, bucket_name, metrics_key)
        print("\n全体メトリクス:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # ローカル保存はオプションに変更（ディスク容量節約のため）
    save_local = os.getenv('SAVE_LOCAL', 'false').lower() == 'true'
    if save_local:
        os.makedirs("output", exist_ok=True)
        if not df_pred_all.empty:
            local_pred_path = f"output/predictions_{timestamp}.parquet"
            df_pred_all.to_parquet(local_pred_path, index=False)
            print(f"\nローカル保存: {local_pred_path}")
    else:
        print("\nローカル保存はスキップ（SAVE_LOCAL=true で有効化）")

    print("\n" + "=" * 50)
    print("学習・予測処理完了")
    print("=" * 50)

    return df_pred_all, metrics

if __name__ == "__main__":
    df_pred, metrics = main()