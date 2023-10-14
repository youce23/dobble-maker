# ビルド手順書

GUI 版 dobble-maker (dobble_maker_gui.py) のビルド手順のまとめ (cx_Freeze 使用[^about_other])

[^about_other]: nuitka は exe の生成に失敗（`galois`が使っている`numba`が主原因？）、PyInstaller はウィルスチェックの誤検知に対処できなかった（bootloader のリビルド、onefile 除外を検証）ため cx_Freeze を選択

動作確認環境

- OS: Windows 10 Home
- 仮想環境: pipenv 2023.8.26 + 同梱の Pipfile, Pipfile.lock
  - `.venv`フォルダに仮想環境が構築されているものとする

## exe の生成

### バージョンの更新

`setup_cx_freeze.py`を更新する

### ビルドの実行

1. `build_cx_freeze.bat`を実行
   1. 軽量化のための除外パッケージ指定の手順は`setup_cx_freeze.py`のコメントを参照
2. 成功すると`dobble-maker\release`に`dobble_maker_gui.exe`が生成される

## ライセンスについて

外部パッケージのライセンスは、`release\lib\library.zip`以下の`*.dist-info`フォルダに格納される

- 仮想環境下に入っている外部パッケージは、ビルド時に除外指定をしていてもライセンスが格納される
