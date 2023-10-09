# ビルド手順書

GUI 版 dobble-maker (dobble_maker_gui.py) のビルド手順のまとめ (PyInstaller 使用[^about_nuitka])

[^about_nuitka]: nuitka も試したが exe の生成に失敗したので PyInstaller にした（`galois`が使っている`numba`が主原因？）

動作確認環境

- OS: Windows 10 Home
- 仮想環境: pipenv 2023.8.26 + 同梱の Pipfile, Pipfile.lock
  - `.venv`フォルダに仮想環境が構築されているものとする

## Bootloader のリビルド + PyInstaller のインストール

PyInstaller をそのまま使うと実行ファイルに対してウィルス検知され exe が削除される場合がある。
これは PyInstaller に内包される Bootloader を再コンパイルすると回避できる。

MinGW (gcc) あるいは Visual Studio C++ コンパイラ (msvc) などでコンパイルできるが、MinGW では後述する`python ./waf distclean all`で gcc が認識されなかったため VS C++ コンパイラ を使う（PyInstaller 公式でも MinGW ではなく[vc コンパイラを使っている](https://www.pyinstaller.org/en/stable/bootloader-building.html#building-for-windows)）

### VisualStudio C++ コンパイラ のインストール

[公式](https://www.pyinstaller.org/en/stable/bootloader-building.html#build-using-visual-studio-c)ではパッケージマネージャとして`Chocolatey`を使う方法が紹介されていたが、vcbuildtools のインストールに失敗したため以下の方法で行う

1. [Build Tools for Visual Studio](https://visualstudio.microsoft.com/ja/downloads/)をダウンロード、実行
2. `C++によるデスクトップ開発`にチェックを入れてインストール

   ![](readme_images/build_tools_conf.png)

### PyInstaller を git clone

`.venv\Lib`に PyInstaller のソースを clone

【追記】 **PyInstaller は普通に`pipenv install pyinstaller`で入れて、任意のフォルダでビルドした`run.exe, run_d.exe, runw.exe, runw_d.exe`を`Lib\site-packages\PyInstaller\bootloader\{OS}`に手動コピーする方が良いのかもしれない**

```cmd
git clone https://github.com/pyinstaller/pyinstaller
```

### Bootloader をリビルド

`.venv\Lib\site-packages\pyinstaller\bootloader`で以下を実行

```cmd
python ./waf distclean all
```

### PyInstaller をインストール

1. 仮想環境`.venv`を有効化
   1. `dobble-maker\.venv`に仮想環境があるなら、`dobble-maker`直下で`pipenv shell`を実行
2. `.venv`があるフォルダ (`dobble-maker`) をカレントとして以下を実行
   ```cmd
   pipenv install -e .venv\Lib\site-packages\pyinstaller
   ```
   これにより、仮想環境にローカルビルドした Bootloader の PyInstaller がインストールされる

## exe の生成

1. `build.bat`を実行
2. 成功すると``に`dobble_maker_gui.exe`が生成される