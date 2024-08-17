"""
cx_Freeze 用セットアップファイル
"""

import sys

from cx_Freeze import Executable, setup

# exeのバージョン番号
VERSION = "0.7.0"

build_exe_options = {
    # 取り込みたいパッケージ
    # "packages", "excludes"を全て空にして実行した時に不足するものを追加する
    "packages": [
        "galois",
    ],
    # 除外したいパッケージ
    # "packages"だけ指定してexeの生成に成功したら、
    # lib以下にあるパッケージから不要なパッケージを選択し除外する。
    #
    # 以下の理由から、あまり減らしすぎない方が良い
    # * 実行時にエラーが出ない不具合が生じることがある
    # * もともと仮想環境が構築されていて余計なパッケージがほとんどなければサイズ削減効果が少ない
    # * cx_Freezeが収集するライセンスはここでの指定に関わらず、インストール済みの全ての外部パッケージが対象
    "excludes": [
        "asyncio",
        "colorama",
        "contourpy",
        "fontTools",
        "http",
        "lxml",
        "pikepdf",
        "setuptools",
        "test",
        "unittest",
        "wheel",
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
