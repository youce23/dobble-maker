"""
cx_Freeze 用セットアップファイル
"""

import sys

from cx_Freeze import Executable, setup

# exeのバージョン番号
VERSION = "0.1.2"

build_exe_options = {
    # 取り込みたいパッケージ
    # "packages", "excludes"を全て空にして実行した時に不足するものを追加する
    "packages": [
        "galois",
    ],
    # 除外したいパッケージ
    # "packages"だけ指定してexeの生成に成功したら、
    # lib以下にある全フォルダ名をここで指定し、ビルドに失敗したら1個ずつ除外する
    # さらにビルドが成功したら、記載済みにも関わらずフォルダが残っているものは指定が無効ということなので除外する
    "excludes": [
        "asyncio",
        "colorama",
        "concurrent",
        "contourpy",
        "distutils",
        "fontTools",
        "http",
        "lib2to3",
        "lxml",
        "multiprocessing",
        "pikepdf",
        "pydoc_data",
        "setuptools",
        "test",
        "unittest",
        "wheel",
        "xmlrpc",
        "yaml",
    ],
}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    # アプリの名前
    name="dobble-maker",
    version=VERSION,
    # アプリの説明
    description="dobble-maker",
    options={
        "build_exe": build_exe_options,
    },
    executables=[
        Executable(
            # 実行ファイル名
            script="dobble_maker_gui.py",
            base=base,
        ),
    ],
)
