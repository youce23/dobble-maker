# dobble-maker

ボードゲーム ドブル (Dobble / Spot it!) の Python 実装です。

ドブルは全 55 枚のカードで構成されています。
各カードには 8 個ずつの画像（シンボル）が描かれていて、そのうちどの 2 枚を選んでも、必ず 1 個のシンボルが共通して存在する構成となっています。

dobble-maker では、任意の画像、カード当たりのシンボル数でカードデッキを構築し、印刷用の PDF を生成します。

# できること

- カードに配置するシンボルに好きな画像を指定可能
- カード 1 枚当たりのシンボル数を任意に指定可能
- カードのサイズを円、または長方形で任意サイズを指定可能
- カード内に程よく均等かつランダムに配置
- カード毎の画像、及び任意サイズの PDF ファイルを出力
- シンボル一覧のカード作成

**カード 1 枚当たりのシンボル数`n=4`で円形の場合に、A4 指定で出力される PDF の 1 ページ目**

<img alt="n=4の場合" src=readme_images/card_4.png width=300px>

**`n=5`の場合**

<img alt="n=5の場合" src=readme_images/card_5.png width=300px>

**`n=5`で長方形（トランプサイズ）の場合**

<img alt="n=5かつトランプサイズの場合" src=readme_images/card_5_trump.png width=300px>

**シンボル一覧のカード**

<img alt="シンボル一覧" src=readme_images/thumbnails.png width=300px>

# 使い方

## ツールの入手と実行

1. [Release - dobble-maker](https://github.com/youce23/dobble-maker/releases) から `dobble_maker_gui_v*.zip` をダウンロード
2. zip ファイルを任意のフォルダに解凍
3. 解凍されたフォルダ内の`dobble_maker_gui.exe`を実行

## パラメータの説明

`dobble_maker_gui.exe`を実行するとパラメータを入力する画面が表示されるので、入力後、`計算実行`ボタンで画像が生成されます。

以下の通りにパラメータを指定すると、同梱の`samples`[^sample_source]を入力として`output`にカード毎の画像、シンボル一覧カードの画像、 PDF ファイル（A4 縦）が生成されます。

[^sample_source]: サンプル画像の入手元: [いらすとや](https://www.irasutoya.com/)

![](readme_images/gui.png)

- 入力画像フォルダ
  - カードに描画するシンボル画像 (JPG または PNG) が保存されたフォルダを指定
    - `カード当たりのシンボル数` を `n` とした場合に、 `n * (n - 1) + 1` 個以上の画像が必要なので、あらかじめ準備しておく
  - 画像 の背景は白 `(255, 255, 255)` あるいは透過であること
- 出力フォルダ
  - カード画像や PDF ファイルを保存するフォルダを指定
- 画像シャッフル
  - チェックを入れると、入力画像フォルダにある画像から使用する画像をランダム選択する
- カード当たりのシンボル数
  - カード 1 枚に描画するシンボル数を指定
    - `2 あるいは 素数の累乗 + 1`であることが条件
  - 例えばドブルキッズは`n = 6`、通常のドブルは`n = 8`で構成されている
- カードの形
  - 生成するカードの形状を指定
- カードサイズ
  - 印刷するカードのサイズを mm で指定
  - 計算時間が画像サイズに比例するため、動作確認やパラメータ調整時は小さめ（30mm など）に設定すると良い
- ページサイズ
  - PDF ファイルの紙サイズ mm で指定
  - 例えば A4(縦)なら (幅 210mm, 高さ 297mm)、A3(横)なら(幅 420mm, 高さ 297mm)を指定する
- レイアウト調整
  - 以下は初期値を基準に、1 か所だけ値を変えて生成、を繰り返して調整するのが良い
  - レイアウト均一性
    - カードに描画するシンボルのサイズ及び配置の均一性を調整するパラメータ
    - 値が小さいとサイズがバラバラで位置がランダムに、大きいと均一かつ整列されやすくなる
  - シンボルサイズ比調整
    - カードに描画するシンボル間のサイズ比を調整するパラメータ
    - 0 では全てのシンボルのサイズが等しく、大きくするとシンボル間のサイズに差が生じやすくなる
- 乱数 seed
  - カードのシンボル配置や画像シャッフルで選択される画像を別パターンにしたい場合、この値を変更する
- シンボル一覧
  - チェックを入れるとシンボル一覧のカード画像を作成できる
  - 入力ファイル
    - 「ファイル名」、「名前」列を持つ`*.xlsx`または`*.csv`を指定
    - 「ファイル名」には「入力画像フォルダ」に画像の拡張子を除くファイル名を記載する
    - 「名前」には表示したいテキストを記載する
      - `\n`と書くと改行される
  - 行数、列数
    - カードに描画するシンボル一覧の行・列数を指定
    - 「名前」が長いと文字が小さくなりやすい。その場合、列数を減らすとシンボルあたりの画像幅が大きくなり、文字サイズを大きくできる。
  - 画像内の最大文字高さ比
    - 文字を大きくしたい場合は値を大きくする。特に改行を含む「名前」を大きくしたい場合にはここで調整する。
  - シンボルサイズ調整値
    - シンボル画像周囲の余白量を調整するパラメータ
    - 「カードの形」が「円」の場合、カードの端に画像が描画されないことがある。その場合はこの値を大きくすることで、シンボル画像が小さくなり、カード端に描画されやすくなる。
