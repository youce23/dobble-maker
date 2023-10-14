# ビルド手順書

GUI 版 dobble-maker (dobble_maker_gui.py) のビルド手順のまとめ (PyInstaller 使用[^about_nuitka])

[^about_nuitka]: nuitka も試したが exe の生成に失敗したので PyInstaller にした（`galois`が使っている`numba`が主原因？）

動作確認環境

- OS: Windows 10 Home
- 仮想環境: pipenv 2023.8.26 + 同梱の Pipfile, Pipfile.lock
  - `.venv`フォルダに仮想環境が構築されているものとする

## exe の生成

### バージョンの更新

`version.yaml`を更新する

### ビルドの実行

1. `build.bat`を実行
   1. `dobble-maker`直下で`pipenv run pip list`でインストール済みパッケージを確認し、不要なものは`exclude-module`で追加する（軽量化、ライセンスの整理が狙い）
2. 成功すると`dobble-maker\release`に`dobble_maker_gui.exe`が生成される

## ライセンスについて

外部パッケージのライセンスは以下の手順で`dobble-maker\release\LICENSES`に追加する

1. `dobble-maker`直下で`pipenv run pip list`を実行しインストール済みパッケージの一覧を確認
2. `build.bat`で`exclude-module`指定のパッケージは exe に含まれないためこれらを上記の一覧から除外
3. 残ったパッケージが exe に内包されるので、`.venv\Lib\site-packages`から該当パッケージのライセンスファイルを`release\LICENSES`以下にコピーする
