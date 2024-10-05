"""3枚に1つのシンボルが共通するデッキを符号理論に基づき生成

以下の手順で構築したものを関数でまとめた。

1. 条件に合う符号(code)を Code Tables で検索
    - Code Tables: Bounds on the parameters of various types of codes
        - https://www.codetables.de/ の "Linear Codes"
    - 「d枚に1つのシンボルが共通」するデッキにする場合、[n, k, d]_GF(q)は[n, n-(d+1), d+1]_GF(q)を満たす必要がある
    - ここではd = 3なので、上記サイトより[n, n-4, 4]_GF(q)の符号を検索する
        - "Complete tables with bounds on linear codes [n,k,d] over GF(q)"というコンテンツで、
        GF(q)毎に全てのn, k, dのセットがテーブル形式で一覧できるので、ここから探すのが良い
2. 1で見つけた算出方法は数値計算ソフト MAGMA を使うことを前提とているため、MAGMA calculatorで実装する
    - MAGMA calculator
        - http://magma.maths.usyd.edu.au/calc/
    - 以下の手順で実装するとラク
        - ChatGPT あるいは Claude などで「以下をMAGMAで実装してください。{1のConstructionを貼り付け}」としてプログラムを生成
        - MAGMA calculator に張り付けて実行し、エラーが出たら再度 ChatGPT / Claude に質問
        - 適宜 MAGMA のドキュメントも参照しながら修正する
            - http://magma.maths.usyd.edu.au/magma/handbook/
3. 上記で符号の 生成行列 G 及び 検査行列 H を生成
4. 検査行列 H からデッキを構築する
"""

import statistics
import sys
from itertools import combinations
from typing import Final

import galois
import numpy as np
from tqdm import tqdm

# cx_Freezeでbase=Win32GUIでビルドされた場合、標準出力が空になりtqdmが使えないため判定
disable_tqdm: Final[bool] = (not sys.stdout) or (not sys.stderr)


def _shortening_last_n(
    G: galois.FieldArray, H: galois.FieldArray, n: int
) -> tuple[galois.FieldArray, galois.FieldArray]:
    """生成行列 G, 検査行列 H に対して ShortenCode(*, {末尾側のn個}) した結果を返す

    NOTE:
    - 本関数は、ShortenCode(*, {末尾側のn個})の処理を完全に再現できていない
    - 以下の有効・無効を確認済み
        - generate_code_26_22_4_GF5, generate_code_17_13_4_GF4 に対してのみ有効であることを確認済み
        - generate_code_10_6_4_GF3 に対しては期待した結果が得られないことを確認済み
    """
    G_shorten = G[n:, n:]
    H_shorten = H[:, :-n]

    assert isinstance(G_shorten, galois.FieldArray)
    assert isinstance(H_shorten, galois.FieldArray)

    return G_shorten, H_shorten


def generate_code_7_3_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[7, 3, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=7&k=3
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {8 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 19)

    return G, H


def generate_code_8_4_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[8, 4, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=8&k=4
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {9 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 18)

    return G, H


def generate_code_9_5_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[9, 5, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=9&k=5
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {10 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 17)

    return G, H


def generate_code_10_6_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[10, 6, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=10&k=6
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {11 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 16)

    return G, H


def generate_code_11_7_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[11, 7, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=11&k=7
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {12 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 15)

    return G, H


def generate_code_12_8_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[12, 8, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=12&k=8
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {13 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 14)

    return G, H


def generate_code_13_9_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[13, 9, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=13&k=9
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {14 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 13)

    return G, H


def generate_code_14_10_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[14, 10, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=14&k=10
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {15 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 12)

    return G, H


def generate_code_15_11_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[15, 11, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=15&k=11
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {16 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 11)

    return G, H


def generate_code_16_12_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[16, 12, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=16&k=12
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {17 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 10)

    return G, H


def generate_code_17_13_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[17, 13, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=17&k=13
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {18 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 9)

    return G, H


def generate_code_18_14_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[18, 14, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=18&k=14
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {19 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 8)

    return G, H


def generate_code_19_15_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[19, 15, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=19&k=15
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {20 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 7)

    return G, H


def generate_code_20_16_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[20, 16, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=20&k=16
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {21 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 6)

    return G, H


def generate_code_21_17_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[21, 17, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=21&k=17
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {22 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 5)

    return G, H


def generate_code_22_18_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[22, 18, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=22&k=18
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {23 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 4)

    return G, H


def generate_code_23_19_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[23, 19, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=23&k=19
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {24 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 3)

    return G, H


def generate_code_24_20_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[24, 20, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=24&k=20
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {25 .. 26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 2)

    return G, H


def generate_code_25_21_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[25, 21, 4], GF(5) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=5&n=25&k=21
        - generate_code_26_22_4_GF5 の MAGMA 実装で C を求めた後に ShortenCode(C, {26}) を追加する
    """
    G, H = generate_code_26_22_4_GF5()
    G, H = _shortening_last_n(G, H, 1)

    return G, H


def generate_code_26_22_4_GF5() -> tuple[galois.FieldArray, galois.FieldArray]:
    """
    [26, 22, 4], GF(5) 符号を表示

    算出方法 <https://www.codetables.de/BKLC/BKLC.php?q=5&n=26&k=22>

    ```
    Construction of a linear code [26,22,4] over GF(5):
    [1]:
        [26, 22, 4] Constacyclic by 3 Linear Code over GF(5)
        ConstaCyclicCode generated by 4*x^25 + 3*x^24 + 2*x^23 + 3*x^22 + 1 with shift constant 3
    ```

    実装

    ```MAGMA
    // 変数定義
    q := 5;  // 位数
    F<x> := PolynomialRing(GF(q));  // GF(q)上の多項式環を定義
    g := 4*x^25 + 3*x^24 + 2*x^23 + 3*x^22 + 1;  // 生成多項式
    shift := GF(q)!3;  // シフト定数
    n := 26;  // 符号長

    // 定数巡回符号を構築
    C := ConstaCyclicCode(n, g, shift);

    // 符号の情報を表示
    print "Code C:";
    print C;

    // 符号のパラメータを表示 (Parameters(C)でエラーが生じたため個別に表示)
    print "Parameters of C:";
    print "Length:", Length(C);
    print "Dimension:", Dimension(C);
    print "Minimum Distance:", MinimumDistance(C);

    // 生成行列, 検査行列を生成し表示
    G := GeneratorMatrix(C);
    H := ParityCheckMatrix(C);
    print "Generator matrix of C:";
    print G;
    print "Parity check matrix of C:";
    print H;

    // 情報表示
    print "Dimension of C:", Dimension(C);  // 次元
    print "Minimum distance of C:", MinimumDistance(C);  // 最小距離

    // 生成行列と検査行列の積が0になることを確認
    print "G * Transpose(H) = 0:", G * Transpose(H) eq ZeroMatrix(GF(q), Nrows(G), Nrows(H));
    ```
    """
    gf = galois.GF(5)
    G = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 3, 4],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0, 2],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 4],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 4, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 3, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 4, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 2, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 3, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 4, 2, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 4, 2, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 3, 4],
    ]
    H = [
        [1, 0, 0, 0, 1, 3, 0, 2, 4, 2, 3, 4, 2, 4, 3, 0, 1, 4, 1, 1, 1, 2, 2, 3, 0, 3],
        [0, 1, 0, 0, 4, 3, 3, 3, 3, 2, 4, 4, 2, 3, 1, 3, 4, 2, 3, 0, 0, 4, 0, 4, 3, 2],
        [0, 0, 1, 0, 1, 2, 3, 0, 2, 0, 0, 3, 1, 1, 1, 1, 4, 3, 3, 4, 1, 2, 1, 3, 4, 1],
        [0, 0, 0, 1, 3, 0, 2, 4, 2, 3, 4, 2, 4, 3, 0, 1, 4, 1, 1, 1, 2, 2, 3, 0, 3, 3],
    ]

    return gf(G), gf(H)


def generate_code_6_2_4_GF4() -> tuple[galois.FieldArray, galois.FieldArray]:
    """
    [6, 2, 4], GF(4) 符号を表示

    算出方法 <https://www.codetables.de/BKLC/BKLC.php?q=4&n=6&k=2>

    ```
    Construction of a linear code [6,2,4] over GF(4):
    [1]:
        [6, 3, 4] Linear Code over GF(2^2)
        Extend the QRCode over GF(4)of length 5
    [2]:
        [6, 2, 4] Linear Code over GF(2^2)
        Subcode of [1]
    ```

    実装
    - 以下を実行すると、符号内に"w", "w^2"が出現するが、それぞれ"2", "3"と読み替えれば良い

    ```MAGMA
    // 変数定義
    q := 2^2;  // 位数
    F<w> := GF(q);  // 有限体

    // Step 1: GF(4)上の長さ5のQR符号を構築
    QR5 := QRCode(F, 5);

    // Step 2: QR符号を拡張して[6,3,4]符号を得る
    ExtendedQR5 := ExtendCode(QR5);

    // 拡張されたQR符号の情報を表示
    print "Extended QR Code:";
    print ExtendedQR5;
    print "Length:", Length(ExtendedQR5);
    print "Dimension:", Dimension(ExtendedQR5);
    print "Minimum Distance:", MinimumDistance(ExtendedQR5);

    // Step 3: [6,2,4]部分符号を構築
    // 生成行列から2つの行を選んで新しい符号を作る
    G := GeneratorMatrix(ExtendedQR5);
    SubG := VerticalJoin(G[1], G[2]);  // 最初の2行を選択
    FinalCode := LinearCode(SubG);

    // 最終的な[6,2,4]符号の情報を表示
    print "\nFinal [6,2,4] Code:";
    print FinalCode;
    print "Length:", Length(FinalCode);
    print "Dimension:", Dimension(FinalCode);
    print "Minimum Distance:", MinimumDistance(FinalCode);

    // 生成行列, 検査行列を生成
    G := GeneratorMatrix(FinalCode);
    H := ParityCheckMatrix(FinalCode);
    print "Generator Matrix of [6,2,4] Code:";
    print G;
    print "Parity check of [6,2,4] Code:";
    print H;
    ```
    """
    gf = galois.GF(4)
    G = [
        [1, 0, 0, 1, 3, 3],
        [0, 1, 0, 3, 3, 1],
    ]
    H = [
        [1, 0, 0, 0, 1, 3],
        [0, 1, 0, 0, 3, 3],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 3, 1],
    ]

    return gf(G), gf(H)


def generate_code_7_3_4_GF4() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[7, 3, 4], GF(4) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=4&n=7&k=3
        - generate_code_17_13_4_GF4 に対して ShortenCode する (generate_code_*_GF5の各関数を参照)
    """
    G, H = generate_code_17_13_4_GF4()
    G, H = _shortening_last_n(G, H, 10)

    return G, H


def generate_code_8_4_4_GF4() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[8, 4, 4], GF(4) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=4&n=8&k=4
        - generate_code_17_13_4_GF4 に対して ShortenCode する (generate_code_*_GF5の各関数を参照)
    """
    G, H = generate_code_17_13_4_GF4()
    G, H = _shortening_last_n(G, H, 9)

    return G, H


def generate_code_9_5_4_GF4() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[9, 5, 4], GF(4) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=4&n=9&k=5
        - generate_code_17_13_4_GF4 に対して ShortenCode する (generate_code_*_GF5の各関数を参照)
    """
    G, H = generate_code_17_13_4_GF4()
    G, H = _shortening_last_n(G, H, 8)

    return G, H


def generate_code_10_6_4_GF4() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[10, 6, 4], GF(4) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=4&n=10&k=6
        - generate_code_17_13_4_GF4 に対して ShortenCode する (generate_code_*_GF5の各関数を参照)
    """
    G, H = generate_code_17_13_4_GF4()
    G, H = _shortening_last_n(G, H, 7)

    return G, H


def generate_code_11_7_4_GF4() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[11, 7, 4], GF(4) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=4&n=11&k=7
        - generate_code_17_13_4_GF4 に対して ShortenCode する (generate_code_*_GF5の各関数を参照)
    """
    G, H = generate_code_17_13_4_GF4()
    G, H = _shortening_last_n(G, H, 6)

    return G, H


def generate_code_12_8_4_GF4() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[12, 8, 4], GF(4) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=4&n=12&k=8
        - generate_code_17_13_4_GF4 に対して ShortenCode する (generate_code_*_GF5の各関数を参照)
    """
    G, H = generate_code_17_13_4_GF4()
    G, H = _shortening_last_n(G, H, 5)

    return G, H


def generate_code_13_9_4_GF4() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[13, 9, 4], GF(4) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=4&n=13&k=9
        - generate_code_17_13_4_GF4 に対して ShortenCode する (generate_code_*_GF5の各関数を参照)
    """
    G, H = generate_code_17_13_4_GF4()
    G, H = _shortening_last_n(G, H, 4)

    return G, H


def generate_code_14_10_4_GF4() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[14, 10, 4], GF(4) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=4&n=14&k=10
        - generate_code_17_13_4_GF4 に対して ShortenCode する (generate_code_*_GF5の各関数を参照)
    """
    G, H = generate_code_17_13_4_GF4()
    G, H = _shortening_last_n(G, H, 3)

    return G, H


def generate_code_15_11_4_GF4() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[15, 11 4], GF(4) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=4&n=15&k=11
        - generate_code_17_13_4_GF4 に対して ShortenCode する (generate_code_*_GF5の各関数を参照)
    """
    G, H = generate_code_17_13_4_GF4()
    G, H = _shortening_last_n(G, H, 2)

    return G, H


def generate_code_16_12_4_GF4() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[16, 12, 4], GF(4) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=4&n=16&k=12
        - generate_code_17_13_4_GF4 に対して ShortenCode する (generate_code_*_GF5の各関数を参照)
    """
    G, H = generate_code_17_13_4_GF4()
    G, H = _shortening_last_n(G, H, 1)

    return G, H


def generate_code_17_13_4_GF4() -> tuple[galois.FieldArray, galois.FieldArray]:
    """
    [17, 13, 4], GF(4) 符号を表示

    算出方法 <https://www.codetables.de/BKLC/BKLC.php?q=4&n=17&k=13>

    ```
    Construction of a linear code [17,13,4] over GF(4):
    [1]:
        [17, 13, 4] "BCH code (d = 3, b = 10)" Linear Code over GF(2^2)
        BCHCode with parameters 17 3 10
    ```

    実装
    - 以下を実行すると、符号内に"w", "w^2"が出現するが、それぞれ"2", "3"と読み替えれば良い

    ```MAGMA
    // 変数定義
    q := 2^2;  // 位数
    F<w> := GF(2^2);  // 有限体
    n := 17;  // 符号長
    d := 3;   // 設計距離
    b := 10;  // 開始指数

    // BCH符号を構築
    C := BCHCode(F, n, d, b);

    // 符号の情報を表示
    print "BCH Code C:";
    print C;

    // 符号のパラメータを表示
    print "Parameters of C:";
    print "Length:", Length(C);
    print "Dimension:", Dimension(C);
    print "Minimum Distance:", MinimumDistance(C);

    // 生成行列, 検査行列を生成し表示
    G := GeneratorMatrix(C);
    H := ParityCheckMatrix(C);
    print "Generator matrix of C:";
    print G;
    print "Parity check matrix of C:";
    print H;

    // 生成行列と検査行列の積が0になることを確認
    print "G * Transpose(H) = 0:", G * Transpose(H) eq ZeroMatrix(GF(q), Nrows(G), Nrows(H));
    ```
    """
    gf = galois.GF(4)
    G = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 3],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 1, 1, 3],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 3, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 3, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 3, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 1, 1],
    ]
    H = [
        [1, 0, 0, 0, 1, 1, 2, 0, 1, 2, 3, 3, 2, 1, 0, 2, 1],
        [0, 1, 0, 0, 1, 0, 3, 2, 1, 3, 1, 0, 1, 3, 1, 2, 3],
        [0, 0, 1, 0, 3, 2, 1, 3, 1, 0, 1, 3, 1, 2, 3, 0, 1],
        [0, 0, 0, 1, 1, 2, 0, 1, 2, 3, 3, 2, 1, 0, 2, 1, 1],
    ]

    return gf(G), gf(H)


def generate_code_6_2_4_GF3() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[6, 2, 4], GF(3) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=3&n=6&k=2
        - generate_code_10_6_4_GF3 に対して, ShortenCode(C, {7 .. 10}) する
    """
    gf = galois.GF(3)
    G = [
        [1, 0, 2, 1, 2, 2],
        [0, 1, 2, 1, 1, 0],
    ]

    H = [
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 2, 1],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 2, 2],
    ]

    return gf(G), gf(H)


def generate_code_7_3_4_GF3() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[7, 3, 4], GF(3) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=3&n=7&k=3
        - generate_code_10_6_4_GF3 に対して, ShortenCode(C, {8 .. 10}) する
    """
    gf = galois.GF(3)
    G = [
        [1, 0, 0, 2, 2, 1, 2],
        [0, 1, 0, 2, 1, 2, 2],
        [0, 0, 1, 1, 0, 2, 2],
    ]

    H = [
        [1, 0, 0, 0, 0, 1, 2],
        [0, 1, 0, 0, 2, 1, 2],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 2, 2, 2],
    ]

    return gf(G), gf(H)


def generate_code_8_4_4_GF3() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[8, 4, 4], GF(3) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=3&n=8&k=4
        - generate_code_10_6_4_GF3 に対して, ShortenCode(C, {9 .. 10}) する
    """
    gf = galois.GF(3)
    G = [
        [1, 0, 0, 0, 0, 2, 2, 1],
        [0, 1, 0, 0, 2, 0, 2, 1],
        [0, 0, 1, 0, 2, 1, 2, 2],
        [0, 0, 0, 1, 1, 1, 0, 1],
    ]

    H = [
        [1, 0, 0, 0, 0, 1, 2, 2],
        [0, 1, 0, 0, 2, 1, 2, 0],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 2, 2, 2, 1],
    ]

    return gf(G), gf(H)


def generate_code_9_5_4_GF3() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[9, 5, 4], GF(3) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=3&n=9&k=5
        - generate_code_10_6_4_GF3 に対して, ShortenCode(C, {10}) する
    """
    gf = galois.GF(3)
    G = [
        [1, 0, 0, 0, 0, 2, 2, 1, 0],
        [0, 1, 0, 0, 0, 2, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 2, 1, 2, 2],
        [0, 0, 0, 0, 1, 2, 2, 2, 1],
    ]

    H = [
        [1, 0, 0, 0, 0, 1, 2, 2, 2],
        [0, 1, 0, 0, 2, 1, 2, 0, 1],
        [0, 0, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 2, 2, 2, 1, 0],
    ]

    return gf(G), gf(H)


def generate_code_10_6_4_GF3() -> tuple[galois.FieldArray, galois.FieldArray]:
    """
    [10, 6, 4], GF(3) 符号を表示

    算出方法 <https://www.codetables.de/BKLC/BKLC.php?q=3&n=10&k=6>

    ```
    Construction of a linear code [10,6,4] over GF(3):
    [1]:
        [12, 6, 6] Linear Code over GF(3)
        Extend the QRCode over GF(3)of length 11
    [2]:
        [10, 6, 4] Linear Code over GF(3)
        Puncturing of [1] at { 11 .. 12 }
    ```

    実装

    ```MAGMA
    // GF(3)を定義
    q := 3;
    F := GF(q);

    // Step 1: GF(3)上の長さ11のQR符号を構築
    n := 11;
    QR11 := QRCode(F, n);

    // Step 2: QR符号を拡張して[12,6,6]符号を得る
    ExtendedQR11 := ExtendCode(QR11);

    // 拡張されたQR符号の情報を表示
    print "Extended QR Code:";
    print ExtendedQR11;
    print "Length:", Length(ExtendedQR11);
    print "Dimension:", Dimension(ExtendedQR11);
    print "Minimum Distance:", MinimumDistance(ExtendedQR11);

    // Step 3: [12,6,6]符号を11と12の位置でpuncturingして[10,6,4]符号を得る
    FinalCode := PunctureCode(ExtendedQR11, {11, 12});

    // 最終的な[10,6,4]符号の情報を表示
    print "\nFinal [10,6,4] Code:";
    print FinalCode;
    print "Length:", Length(FinalCode);
    print "Dimension:", Dimension(FinalCode);
    print "Minimum Distance:", MinimumDistance(FinalCode);

    // 生成行列を表示
    G := GeneratorMatrix(FinalCode);
    H := ParityCheckMatrix(FinalCode);
    print "Generator Matrix of [10,6,4] Code:";
    print G;
    print "Parity check matrix of [10,6,4] Code:";
    print H;

    // 生成行列と検査行列の積が0になることを確認
    print "G * Transpose(H) = 0:", G * Transpose(H) eq ZeroMatrix(GF(q), Nrows(G), Nrows(H));
    ```
    """
    gf = galois.GF(3)
    G = [
        [1, 0, 0, 0, 0, 0, 2, 0, 1, 2],
        [0, 1, 0, 0, 0, 0, 1, 2, 2, 2],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 2],
        [0, 0, 0, 0, 1, 0, 2, 1, 2, 2],
        [0, 0, 0, 0, 0, 1, 0, 2, 1, 2],
    ]
    H = [
        [1, 0, 0, 0, 0, 1, 2, 2, 2, 1],
        [0, 1, 0, 0, 2, 1, 2, 0, 1, 2],
        [0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 2, 2, 2, 1, 0, 1],
    ]

    return gf(G), gf(H)


def generate_code_6_2_4_GF2() -> tuple[galois.FieldArray, galois.FieldArray]:
    """
    [6, 2, 4], GF(2) 符号を表示

    算出方法 <https://www.codetables.de/BKLC/BKLC.php?q=2&n=6&k=2>

    ```
    Construction of a linear code [6,2,4] over GF(2):
    [1]:
        [6, 2, 4] Quasicyclic of degree 3 Linear Code over GF(2)
        CordaroWagnerCode of length 6
    ```

    実装

    ```MAGMA
    // Cordaro-Wagner code of length 6 を生成 (Cordaro-Wagner codeのfieldは2固定)
    C := CordaroWagnerCode(6);

    // 生成行列, 検査行列を生成
    G := GeneratorMatrix(C);
    H := ParityCheckMatrix(C);

    print "Generator Matrix:", G;
    print "Parity-Check Matrix:", H;

    // 生成行列と検査行列の積が0になることを確認
    q := 2;
    print "G * Transpose(H) = 0:", G * Transpose(H) eq ZeroMatrix(GF(q), Nrows(G), Nrows(H));
    ```
    """
    gf = galois.GF(2)
    G = [
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1],
    ]
    H = [
        [1, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
    ]

    return gf(G), gf(H)


def generate_code_7_3_4_GF2() -> tuple[galois.FieldArray, galois.FieldArray]:
    """[7, 3, 4], GF(2) 符号を表示

    - 算出方法: https://www.codetables.de/BKLC/BKLC.php?q=2&n=7&k=3
        - generate_code_8_4_4_GF2 に対して, ShortenCode(C, {1})する
    """
    gf = galois.GF(2)
    G = [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ]
    H = [
        [1, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1],
    ]

    return gf(G), gf(H)


def generate_code_8_4_4_GF2() -> tuple[galois.FieldArray, galois.FieldArray]:
    """
    [8, 4, 4], GF(2) 符号を表示

    算出方法 <https://www.codetables.de/BKLC/BKLC.php?q=2&n=8&k=4>

    ```
    Construction of a linear code [8,4,4] over GF(2):
    [1]:
        [4, 1, 4] Cyclic Linear Code over GF(2)
        RepetitionCode of length 4
    [2]:
        [4, 3, 2] Cyclic Linear Code over GF(2)
        Dual of the RepetitionCode of length 4
    [3]:
        [8, 4, 4] Quasicyclic of degree 2 Linear Code over GF(2)
        PlotkinSum of [2] and [1]
    ```

    実装

    ```MAGMA
    // GF(2)を定義
    q := 2;
    F := GF(q);

    // Step 1: 長さ4の繰り返し符号を構築
    C1 := RepetitionCode(F, 4);
    print "C1 (Repetition Code):";
    print C1;
    print "Length:", Length(C1);
    print "Dimension:", Dimension(C1);
    print "Minimum Distance:", MinimumDistance(C1);

    // Step 2: C1の双対符号を構築
    C2 := Dual(C1);
    print "\nC2 (Dual of Repetition Code):";
    print C2;
    print "Length:", Length(C2);
    print "Dimension:", Dimension(C2);
    print "Minimum Distance:", MinimumDistance(C2);

    // Step 3: C1とC2のPlotkin和を計算して[8,4,4]符号を得る
    C3 := PlotkinSum(C2, C1);
    print "\nC3 (Plotkin Sum of C2 and C1):";
    print C3;
    print "Length:", Length(C3);
    print "Dimension:", Dimension(C3);
    print "Minimum Distance:", MinimumDistance(C3);

    // 生成行列を表示
    G := GeneratorMatrix(C3);
    H := ParityCheckMatrix(C3);
    print "Generator Matrix of [8,4,4] Code:";
    print G;
    print "Parity check matrix of [8,4,4] Code:";
    print H;

    // 生成行列と検査行列の積が0になることを確認
    print "G * Transpose(H) = 0:", G * Transpose(H) eq ZeroMatrix(GF(q), Nrows(G), Nrows(H));
    ```
    """
    gf = galois.GF(2)
    G = [
        [1, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ]
    H = [
        [1, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ]

    return gf(G), gf(H)


def monic_normalize(vector: galois.FieldArray) -> galois.FieldArray:
    """有限体上のベクトルを正規化 (モニック化)

    ベクトルの各要素を多項式の係数として、最高次の項の係数を1にする
    - つまり最初に見つけた非ゼロの値を1にする

    Args:
        vector (galois.FieldArray): 入力ベクトル

    Returns:
        np.ndarray: 正規化されたベクトル
    """
    # 最初の非ゼロ要素を見つける
    first_val = next((x for x in vector if x != 0), None)

    if first_val is None:
        return vector  # ゼロベクトル
    elif first_val == 1:
        return vector  # 正規化済み

    # 最初の非ゼロ要素の逆数を計算
    gf = galois.GF(first_val._order)
    inverse = gf(1) / first_val

    # ベクトルをこの逆数で乗算
    normed_vec = vector * inverse

    return normed_vec


def generate_deck_from_parity_check_matrix(H: galois.FieldArray) -> tuple[list[list[int]], int, int, int]:
    """パリティ検査行列に従ってドブルデッキ構築

    以下の制約がある
    - 各カードに記載されるシンボル数が一定ではない場合がある
    - 各シンボルの出現回数は一定ではない場合がある

    Args:
        H: パリティ検査行列

    Returns:
        list[list[int]: デッキ
        int: 全シンボル数
        int: 全カード数
        int: カード1枚当たりのシンボル数 (最頻値)

    """
    q = H._order
    d, k = H.shape

    # Hから(d-1)本の列ベクトルを全組み合わせで選択
    deck: list[set[int]] = [set() for _ in range(k)]  # decks[カードID] = {保有するシンボルID}とする
    all_combi = list(combinations(range(k), d - 1))
    for combi in tqdm(all_combi, disable=disable_tqdm):
        # 選択した列ベクトルが線形独立か確認
        selected = H[:, combi]
        assert isinstance(selected, galois.FieldArray)
        rank = np.linalg.matrix_rank(selected)
        if rank != d - 1:
            continue
        # 線形独立である組について、その補空間のベクトルを算出(必ず次元は1となる)
        orth_vecs = selected.left_null_space()
        assert len(orth_vecs) == 1
        assert isinstance(orth_vecs, galois.FieldArray)
        orth_vec = orth_vecs[0]
        assert isinstance(orth_vec, galois.FieldArray)
        # 正規化
        normalized_orth_vec = monic_normalize(orth_vec)

        # 選択した(d-1)本のベクトル = 選択した(d-1)枚のカード, 補空間 = 共通シンボル, と解釈して
        # 各カードが保有するシンボルを記憶
        symbol_id = sum(int(a) * q**i for i, a in enumerate(normalized_orth_vec))
        for card_id in combi:
            deck[card_id].add(symbol_id)

    # シンボルIDを整理
    all_symbol_ids = set().union(*deck)  # decksの全ての要素をOR結合
    all_symbol_ids = sorted(all_symbol_ids)
    symbol_ids_map = {symbol_id: i for i, symbol_id in enumerate(all_symbol_ids)}
    sorted_deck = [sorted([symbol_ids_map[id] for id in card]) for card in deck]
    sorted_deck = sorted(sorted_deck)

    _n_symbols_in_card = [len(card) for card in deck]
    n_symbols_per_card = statistics.mode(
        _n_symbols_in_card
    )  # カード毎のシンボル数にバラつきがある可能性があるので最頻値
    n_symbols = len(all_symbol_ids)
    n_cards = len(sorted_deck)

    return sorted_deck, n_symbols, n_cards, n_symbols_per_card


def _check_columns_independent_of_parity_check_matrix(H: galois.FieldArray) -> bool:
    """検査行列 H の(d-1)本の列ベクトルが線形独立であるかを確認

    全組み合わせが線形独立ならTrue
    """
    d, n = H.shape

    columns = list(combinations(range(H.shape[1]), d - 1))
    n_independent: int = 0
    for _columns in columns:
        selected = H[:, _columns]
        rank = np.linalg.matrix_rank(selected)
        n_independent += rank == min(selected.shape)
    print(f"Combination({n}, {d-1}) = {len(columns)}個のうち {n_independent} 個の組み合わせが線形独立")

    return n_independent == len(columns)


def get_valid_params() -> list[tuple[int, int]]:
    """make_dobble_deckで処理可能なn_symbols_per_card, n_cardsの組み合わせを取得"""
    params = [
        (30, 26),
        (30, 25),
        (30, 24),
        (30, 23),
        (30, 22),
        (30, 21),
        (30, 20),
        (30, 19),
        (29, 18),
        (29, 17),
        (29, 16),
        (29, 15),
        (28, 14),
        (26, 13),
        (24, 12),
        (22, 11),
        (18, 10),
        (15, 9),
        (13, 8),
        (11, 7),
        (20, 17),
        (20, 16),
        (20, 15),
        (20, 14),
        (19, 13),
        (18, 12),
        (17, 11),
        (15, 10),
        (15, 9),
        (11, 8),
        (11, 7),
        (5, 6),
        (12, 10),
        (12, 9),
        (11, 8),
        (9, 7),
        (8, 6),
        (7, 8),
        (7, 7),
        (6, 6),
    ]

    return params


def make_dobble_deck(n_symbols_per_card: int, n_cards: int) -> tuple[list[list[int]], int]:
    """各カードに記載するシンボルの一覧を生成

    3枚のカードに1つの共通するシンボルが出現する拡張ドブル版

    Args:
        n_symbols_per_card: カード1枚あたりに記載するシンボル数
        n_cards: 全カード数

    Returns:
        list[list[int]]: デッキ (各カードに記載するシンボル番号)
        int: 全シンボル数
    """
    match (n_symbols_per_card, n_cards):
        case (30, 26):
            G, H = generate_code_26_22_4_GF5()
        case (30, 25):
            G, H = generate_code_25_21_4_GF5()
        case (30, 24):
            G, H = generate_code_24_20_4_GF5()
        case (30, 23):
            G, H = generate_code_23_19_4_GF5()
        case (30, 22):
            G, H = generate_code_22_18_4_GF5()
        case (30, 21):
            G, H = generate_code_21_17_4_GF5()
        case (30, 20):
            G, H = generate_code_20_16_4_GF5()
        case (30, 19):
            G, H = generate_code_19_15_4_GF5()
        case (29, 18):
            G, H = generate_code_18_14_4_GF5()
        case (29, 17):
            G, H = generate_code_17_13_4_GF5()
        case (29, 16):
            G, H = generate_code_16_12_4_GF5()
        case (29, 15):
            G, H = generate_code_15_11_4_GF5()
        case (28, 14):
            G, H = generate_code_14_10_4_GF5()
        case (26, 13):
            G, H = generate_code_13_9_4_GF5()
        case (24, 12):
            G, H = generate_code_12_8_4_GF5()
        case (22, 11):
            G, H = generate_code_11_7_4_GF5()
        case (18, 10):
            G, H = generate_code_10_6_4_GF5()
        case (15, 9):
            G, H = generate_code_9_5_4_GF5()
        case (13, 8):
            G, H = generate_code_8_4_4_GF5()
        case (11, 7):
            G, H = generate_code_7_3_4_GF5()
        case (20, 17):
            G, H = generate_code_17_13_4_GF4()
        case (20, 16):
            G, H = generate_code_16_12_4_GF4()
        case (20, 15):
            G, H = generate_code_15_11_4_GF4()
        case (20, 14):
            G, H = generate_code_14_10_4_GF4()
        case (19, 13):
            G, H = generate_code_13_9_4_GF4()
        case (18, 12):
            G, H = generate_code_12_8_4_GF4()
        case (17, 11):
            G, H = generate_code_11_7_4_GF4()
        case (15, 10):
            G, H = generate_code_10_6_4_GF4()
        case (15, 9):
            # NOTE: 同じ条件式があるためここは通らない
            G, H = generate_code_9_5_4_GF4()
        case (11, 8):
            G, H = generate_code_8_4_4_GF4()
        case (11, 7):
            # NOTE: 同じ条件式があるためここは通らない
            G, H = generate_code_7_3_4_GF4()
        case (5, 6):
            # NOTE: 同じ条件式があるためここは通らない
            G, H = generate_code_6_2_4_GF4()
        case (12, 10):
            G, H = generate_code_10_6_4_GF3()
        case (12, 9):
            G, H = generate_code_9_5_4_GF3()
        case (11, 8):
            G, H = generate_code_8_4_4_GF3()
        case (9, 7):
            G, H = generate_code_7_3_4_GF3()
        case (8, 6):
            G, H = generate_code_6_2_4_GF3()
        case (7, 8):
            G, H = generate_code_8_4_4_GF2()
        case (7, 7):
            G, H = generate_code_7_3_4_GF2()
        case (6, 6):
            G, H = generate_code_6_2_4_GF2()
        case _:
            # NOTE: 新たに追加する際は以下で n_symbols_per_card, n_cards を確認する
            # G, H = generate_code_24_20_4_GF5()  # 適切な関数に変更
            # assert np.all(G.dot(H.T) == 0)
            # assert _check_columns_independent_of_parity_check_matrix(H)
            # deck, n_symbols, n_cards, n_symbols_per_card = generate_deck_from_parity_check_matrix(H)
            # print(f"n_symbols_per_card = {n_symbols_per_card}, n_cards = {n_cards}, n_symbols = {n_symbols}")
            raise ValueError(
                f"カード1枚あたりのシンボル数 {n_symbols_per_card} と 全カード数 {n_cards} は所定の値でなければならない"
            )

    # 条件を満たした行列か確認
    n = n_cards
    k = n_cards - 4
    d = 4
    assert G.shape == (k, n)
    assert H.shape == (d, n)
    assert np.linalg.matrix_rank(G) == k
    assert np.linalg.matrix_rank(H) == d
    assert np.all(G.dot(H.T) == 0)
    assert _check_columns_independent_of_parity_check_matrix(H)

    deck, n_symbols, _, _ = generate_deck_from_parity_check_matrix(H)

    return deck, n_symbols


def main() -> None:
    funcs = [
        generate_code_26_22_4_GF5,
        generate_code_25_21_4_GF5,
        generate_code_24_20_4_GF5,
        generate_code_23_19_4_GF5,
        generate_code_22_18_4_GF5,
        generate_code_21_17_4_GF5,
        generate_code_20_16_4_GF5,
        generate_code_19_15_4_GF5,
        generate_code_18_14_4_GF5,
        generate_code_17_13_4_GF5,
        generate_code_16_12_4_GF5,
        generate_code_15_11_4_GF5,
        generate_code_14_10_4_GF5,
        generate_code_13_9_4_GF5,
        generate_code_12_8_4_GF5,
        generate_code_11_7_4_GF5,
        generate_code_10_6_4_GF5,
        generate_code_9_5_4_GF5,
        generate_code_8_4_4_GF5,
        generate_code_7_3_4_GF5,
        generate_code_17_13_4_GF4,
        generate_code_16_12_4_GF4,
        generate_code_15_11_4_GF4,
        generate_code_14_10_4_GF4,
        generate_code_13_9_4_GF4,
        generate_code_12_8_4_GF4,
        generate_code_11_7_4_GF4,
        generate_code_10_6_4_GF4,
        generate_code_9_5_4_GF4,
        generate_code_8_4_4_GF4,
        generate_code_7_3_4_GF4,
        generate_code_6_2_4_GF4,
        generate_code_10_6_4_GF3,
        generate_code_9_5_4_GF3,
        generate_code_8_4_4_GF3,
        generate_code_7_3_4_GF3,
        generate_code_6_2_4_GF3,
        generate_code_8_4_4_GF2,
        generate_code_7_3_4_GF2,
        generate_code_6_2_4_GF2,
    ]
    for func in funcs:
        G, H = func()

        assert np.all(G.dot(H.T) == 0)
        assert _check_columns_independent_of_parity_check_matrix(H)

        deck, n_symbols, n_cards, n_symbols_per_card = generate_deck_from_parity_check_matrix(H)
        print(f"{func.__name__}:")
        print(f"  n_symbols_per_card: {n_symbols_per_card}")
        print(f"  n_cards: {n_cards}")
        print(f"  n_symbols: {n_symbols}")

    return


if __name__ == "__main__":
    main()
