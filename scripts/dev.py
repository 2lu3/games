#!/usr/bin/env python3
"""
開発用スクリプト

srcレイアウトでの開発を容易にするためのヘルパースクリプト
"""

import os
import sys
from pathlib import Path

# srcディレクトリをPythonパスに追加
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    # メインスクリプトを実行
    from utttrlsim.main import main

    main()
