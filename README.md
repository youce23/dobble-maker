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

**カード 1 枚当たりのシンボル数`n=4`で円形の場合に、A4 指定で出力される PDF の 1 ページ目**

<img alt="n=4の場合" src=readme_images/card_4.png width=300px>

**`n=5`の場合**

<img alt="n=5の場合" src=readme_images/card_5.png width=300px>

**`n=5`で長方形（トランプサイズ）の場合**

<img alt="n=5かつトランプサイズの場合" src=readme_images/card_5_trump.png width=300px>

# 環境構築 / 動作確認

dobble-maker を clone し`Pipfile`に従って必要なパッケージをインストールします。

`python dobble_maker.py`を実行し、`output`フォルダにカード毎の画像と`card.pdf`が生成されれば正常動作です。

- `samples`フォルダの画像からシンボル数`n=5`のカード画像が生成されます[^sample_source]

[^sample_source]: サンプル画像の入手元: [いらすとや](https://www.irasutoya.com/)

# 実行手順

任意の画像でドブルデッキを構築する手順です。

## カード 1 枚に描画するシンボル数`n`を決める

`n`は`2 あるいは 素数の累乗 + 1`であることが条件です。

- 2, 3, 4, 5, 6, 8, 9, 10, 12, ...

例えばドブルキッズは`n = 6`、通常のドブルは`n = 8`で構成されています。

## カードに描画する画像`n * (n-1) + 1`枚を準備

カードに描画する好きな PNG 画像`n * (n-1) + 1`枚 を準備します。

画像は、clone してきたフォルダの直下に`images/*.png`, ..., として保存します。

- PNG の背景は白`(255, 255, 255)`あるいは透過である必要があります
- `images`は`samples`と同様の構成になれば OK

## パラメータを修正

clone したフォルダの直下にある`dobble_maker.py`の main にパラメータが定義されているので、適宜修正します。

最低限、以下を確認してください。

- `image_dir`
  - 対象画像を保存したフォルダ`images`に変更してください
- `output_dir`, `pdf_name`
  - 既に実行した結果があるとファイルが上書きされてしまうので注意してください
- `n_symbols_per_card`
  - 先に決めた`n`の値を入れてください

```python
def main():
    # ============
    # パラメータ設定
    # ============
    # ファイル名
    image_dir = "samples"  # 入力画像ディレクトリ
    output_dir = "output"  # 出力画像ディレクトリ
    pdf_name = "card.pdf"  # 出力するPDF名
    # カードの設定
    n_symbols_per_card: int = 5  # カード1枚当たりのシンボル数
    card_img_size = 1500  # カード1枚当たりの画像サイズ (intなら円、(幅, 高さ) なら矩形で作成) [pix]
    card_margin = 20  # カード1枚の余白サイズ [pix]
    layout_method: Literal["random", "voronoi"] = "voronoi"  # random: ランダム配置, voronoi: 重心ボロノイ分割に基づき配置
    # PDFの設定
    dpi = 300  # 解像度
    card_size_mm = 95  # カードの長辺サイズ[mm]
    page_size_mm = (210, 297)  # PDFの(幅, 高さ)[mm]
```

## 実行

`python dobble_maker.py`で実行すると、`output_dir`で指定したフォルダに、カード毎の画像と、`pdf_name`で指定した PDF ファイルが生成されます。
