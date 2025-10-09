#!/usr/bin/env python3
"""
ゼロ補完データの準備実行スクリプト
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.core.prepare.prepare_data_fill_zero import main

if __name__ == "__main__":
    exit(main())