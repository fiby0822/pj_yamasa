#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S3のCSVファイルを確認し、Tableau用に最適化"""

import boto3
import os
import pandas as pd
from io import BytesIO, StringIO
from dotenv import load_dotenv

load_dotenv()

def main():
    s3 = boto3.client('s3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))

    bucket_name = 'fiby-yamasa-prediction'

    # Check and fix CSV files
    csv_files = [
        'models/error_analysis_key_total_latest.csv',
        'models/error_analysis_key_date_latest.csv'
    ]

    for key in csv_files:
        file_name = key.split('/')[-1]
        print(f'\n{"="*50}')
        print(f'ファイル: {file_name}')
        print(f'S3パス: s3://{bucket_name}/{key}')
        print("="*50)

        try:
            # Download file
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            content = obj['Body'].read()

            # Parse CSV
            df = pd.read_csv(BytesIO(content))

            print(f'✓ 正常に読み込めました')
            print(f'  - エンコーディング: UTF-8')
            print(f'  - 行数: {len(df):,}')
            print(f'  - カラム数: {len(df.columns)}')
            print(f'  - カラム名: {", ".join(df.columns)}')

            # データ型を確認
            print(f'\nデータ型:')
            for col in df.columns:
                print(f'  - {col}: {df[col].dtype}')

            # サンプルデータ
            print(f'\nサンプルデータ（最初の3行）:')
            print(df.head(3).to_string())

            # 統計情報
            if 'error_rate_total' in df.columns:
                print(f'\n予測誤差統計:')
                print(f'  - 平均誤差率: {df["error_rate_total"].mean():.2%}')
                print(f'  - 中央誤差率: {df["error_rate_total"].median():.2%}')

            if 'key_mean_abs_err_div_pred_overall' in df.columns:
                print(f'\nkey_mean_abs_err_div_pred_overall統計:')
                print(f'  - 平均: {df["key_mean_abs_err_div_pred_overall"].mean():.4f}')
                print(f'  - 中央値: {df["key_mean_abs_err_div_pred_overall"].median():.4f}')

            # Tableauで読み込みやすい形式で保存し直す（BOM付きUTF-8）
            # BOMを付けることで、Excelなどでも正しく開ける
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_content = csv_buffer.getvalue()

            # BOM付きで再アップロード
            bom_content = '\ufeff' + csv_content  # BOM追加

            new_key = key.replace('_latest.csv', '_tableau_latest.csv')
            s3.put_object(
                Bucket=bucket_name,
                Key=new_key,
                Body=bom_content.encode('utf-8'),
                ContentType='text/csv; charset=utf-8'
            )
            print(f'\n✓ Tableau用ファイル作成: s3://{bucket_name}/{new_key}')

        except Exception as e:
            print(f'✗ エラー: {e}')

    print('\n' + '='*50)
    print('Tableauでの接続方法:')
    print('1. TableauでAmazon S3コネクタを選択')
    print('2. AWS認証情報を入力')
    print('3. バケット: fiby-yamasa-prediction')
    print('4. ファイルパス:')
    print('   - models/error_analysis_key_total_tableau_latest.csv (Material Key毎の集計)')
    print('   - models/error_analysis_key_date_tableau_latest.csv (日次詳細)')
    print('='*50)

if __name__ == '__main__':
    main()