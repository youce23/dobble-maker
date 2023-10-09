import os
import random
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
from typing import Literal

import cv2
import numpy as np

from dobble_maker import (
    images_to_pdf,
    is_valid_n_symbols_per_card,
    layout_images_randomly_wo_overlap,
    load_images,
    make_dobble_deck,
)


class Application(tk.Frame):
    def _select_input_dir(self):
        # 入力フォルダ選択ダイアログ
        dir = self.input_dir.get()
        if not os.path.exists(dir):
            # 存在しないフォルダがテキストボックスに入力されていたら
            # ダイアログではカレントフォルダを初期表示
            dir = os.getcwd()
        dir_name = filedialog.askdirectory(initialdir=dir)
        if dir_name == "":  # キャンセル押下時
            return
        self.input_dir.set(dir_name)

    def _select_output_dir(self):
        # 出力フォルダ選択ダイアログ (処理はself._select_input_dirと同様)
        dir = self.output_dir.get()
        if not os.path.exists(dir):
            dir = os.getcwd()
        dir_name = filedialog.askdirectory(initialdir=dir, mustexist=False)  # NOTE: Windows環境では mustexist=False が効かなかった
        if dir_name == "":  # キャンセル押下時
            return
        self.output_dir.set(dir_name)

    def __init__(self, master=None):
        cur_dir = os.getcwd()

        # Tkinterウィンドウの設定
        super().__init__(master)

        self.master.title("dobble-maker-gui")

        # 入力フォルダ選択ボタン
        self.input_dir = tk.StringVar(value=cur_dir + os.sep + "samples")
        input_dir_label = tk.Label(self.master, text="入力画像フォルダ")
        input_dir_entry = tk.Entry(self.master, width=64, textvariable=self.input_dir)
        input_dir_button = tk.Button(self.master, text="...", command=self._select_input_dir)

        # 出力フォルダ選択ボタン
        self.output_dir = tk.StringVar(value=cur_dir + os.sep + "output")
        output_dir_label = tk.Label(self.master, text="出力フォルダ")
        output_dir_entry = tk.Entry(self.master, width=64, textvariable=self.output_dir)
        output_dir_button = tk.Button(self.master, text="...", command=self._select_output_dir)

        # 入力フォルダの画像シャッフル
        self.shuffle = tk.BooleanVar(value=False)
        shuffle_label = tk.Label(self.master, text="画像シャッフル")
        shuffle_entry = tk.Checkbutton(self.master, variable=self.shuffle)

        # カード当たりのシンボル数を入力するリスト
        n_symbols_vals = [x for x in range(100) if is_valid_n_symbols_per_card(x)]
        self.n_symbols = tk.IntVar(value=5)
        n_symbols_label = tk.Label(self.master, text="カード当たりのシンボル数")
        n_symbols_entry = ttk.Combobox(
            self.master, width=6, height=5, state="readonly", values=n_symbols_vals, textvariable=self.n_symbols
        )

        # カード形状
        self.card_shape = tk.StringVar(value="円")
        card_shape_label = tk.Label(self.master, text="カードの形")
        card_shape_entry = ttk.Combobox(
            self.master, width=6, height=2, state="readonly", values=("円", "長方形"), textvariable=self.card_shape
        )

        # カードサイズ
        card_size_label = tk.Label(self.master, text="カードサイズ (整数)")

        # カード幅 [mm]
        self.card_width = tk.IntVar(value=95)
        card_width_label = tk.Label(self.master, text="幅[mm], 「円」なら直径")
        card_width_entry = tk.Spinbox(self.master, width=6, from_=1, to=1000, increment=1, textvariable=self.card_width)

        # カード高さ [mm]
        self.card_height = tk.IntVar(value=95)
        card_height_label = tk.Label(self.master, text="高さ[mm], 「円」なら無視")
        card_height_entry = tk.Spinbox(
            self.master, width=6, from_=1, to=1000, increment=1, textvariable=self.card_height
        )

        # カード内マージン [mm]
        self.card_margin = tk.IntVar(value=2)
        card_margin_label = tk.Label(self.master, text="余白[mm]")
        card_margin_entry = tk.Spinbox(
            self.master, width=6, from_=0, to=1000, increment=1, textvariable=self.card_margin
        )

        # PDF ページサイズ
        page_size_label = tk.Label(self.master, text="ページサイズ (整数)")

        # ページ幅 [mm]
        self.page_width = tk.IntVar(value=210)
        page_width_label = tk.Label(self.master, text="幅[mm]")
        page_width_entry = tk.Entry(self.master, width=6, textvariable=self.page_width)

        # PDF ページ高さ [mm]
        self.page_height = tk.IntVar(value=297)
        page_height_label = tk.Label(self.master, text="高さ[mm]")
        page_height_entry = tk.Entry(self.master, width=6, textvariable=self.page_height)

        # レイアウトの均一性 (重心ボロノイ分割の反復回数)
        self.cvt_level = tk.IntVar(value=5)
        cvt_level_label = tk.Label(self.master, text="シンボルサイズの均一性 (1から10)")
        cvt_level_entry = tk.Spinbox(
            self.master, state="readonly", width=6, from_=1, to=10, increment=1, textvariable=self.cvt_level
        )

        # seed
        self.seed = tk.IntVar(value=0)
        seed_label = tk.Label(self.master, text="乱数seed (整数)")
        seed_entry = tk.Entry(self.master, width=6, textvariable=self.seed)

        # 計算実行ボタン
        calculate_button = tk.Button(self.master, text="計算実行", command=self._run)

        # ----------------
        # レイアウト (grid)
        # ----------------
        row = 0
        input_dir_label.grid(row=row, column=0, stick=tk.W)
        input_dir_entry.grid(row=row, column=1, stick=tk.W)
        input_dir_button.grid(row=row, column=2, stick=tk.W)
        row += 1

        output_dir_label.grid(row=row, column=0, stick=tk.W)
        output_dir_entry.grid(row=row, column=1, stick=tk.W)
        output_dir_button.grid(row=row, column=2, stick=tk.W)
        row += 1

        shuffle_label.grid(row=row, column=0, stick=tk.W)
        shuffle_entry.grid(row=row, column=1, sticky=tk.W)
        row += 1

        n_symbols_label.grid(row=row, column=0, stick=tk.W)
        n_symbols_entry.grid(row=row, column=1, stick=tk.W)
        row += 1

        card_shape_label.grid(row=row, column=0, stick=tk.W)
        card_shape_entry.grid(row=row, column=1, stick=tk.W)
        row += 1

        card_size_label.grid(row=row, column=0, stick=tk.W, columnspan=3)
        row += 1
        card_width_label.grid(row=row, column=0, stick=tk.E)
        card_width_entry.grid(row=row, column=1, stick=tk.W)
        row += 1
        card_height_label.grid(row=row, column=0, stick=tk.E)
        card_height_entry.grid(row=row, column=1, stick=tk.W)
        row += 1
        card_margin_label.grid(row=row, column=0, stick=tk.E)
        card_margin_entry.grid(row=row, column=1, stick=tk.W)
        row += 1

        page_size_label.grid(row=row, column=0, stick=tk.W, columnspan=3)
        row += 1
        page_width_label.grid(row=row, column=0, stick=tk.E)
        page_width_entry.grid(row=row, column=1, stick=tk.W)
        row += 1
        page_height_label.grid(row=row, column=0, stick=tk.E)
        page_height_entry.grid(row=row, column=1, stick=tk.W)
        row += 1

        cvt_level_label.grid(row=row, column=0, stick=tk.W)
        cvt_level_entry.grid(row=row, column=1, stick=tk.W)
        row += 1

        seed_label.grid(row=row, column=0, stick=tk.W)
        seed_entry.grid(row=row, column=1, stick=tk.W)
        row += 1

        calculate_button.grid(row=row, column=0, pady=5, columnspan=3)
        row += 1

        # アプリケーションの実行
        self.master.mainloop()

    def _error_check_params(self) -> str:
        # パラメータチェック ("readonly"があるパラメータはチェック対象外)
        # 異常があればエラーメッセージ, 無ければ空の文字列を返す
        if not os.path.exists(self.input_dir.get()):
            return "入力フォルダが存在しません"

        try:
            card_width_i = self.card_width.get()
        except Exception:
            return "カード幅は正の整数で入力してください"
        if self.card_shape != "円":
            try:
                card_height_i = self.card_height.get()
            except Exception:
                return "カード高さは正の整数で入力してください"
        else:
            card_height_i = card_width_i
        try:
            card_margin_i = self.card_margin.get()
        except Exception:
            return "カードの余白は正の整数で入力してください"
        try:
            page_width_i = self.page_width.get()
        except Exception:
            return "ページ幅は正の整数で入力してください"
        try:
            page_height_i = self.page_height.get()
        except Exception:
            return "ページ高さは正の整数で入力してください"
        try:
            _ = self.seed.get()
        except Exception:
            return "乱数seedは正の整数で入力してください"

        if page_width_i <= 0:
            return "ページ幅は正の整数で入力してください"
        if page_height_i <= 0:
            return "ページ高さは正の整数で入力してください"
        if not (0 < card_width_i <= page_width_i):
            return "カード幅は 1 以上 ページ幅 以下の整数で入力してください"
        if not (0 < card_height_i <= page_height_i):
            return "カード高さは 1 以上 ページ幅 以下の整数で入力してください"
        if not (0 <= 2 * card_margin_i < min(card_width_i, card_height_i)):
            return "2 * (カードの余白) がカードサイズをオーバーします"

        return ""

    def _run(self):
        err_msg = self._error_check_params()
        if err_msg != "":
            messagebox.showerror("エラー", err_msg)
            return

        # 各パラメータを取得
        DPI = 300  # 解像度
        MM_PER_INCH = 25.4
        pix_per_mm = DPI / MM_PER_INCH  # 1mmあたりのピクセル数

        _card_shape: Literal["円", "長方形"] = self.card_shape.get()

        image_dir = self.input_dir.get()  # 入力画像ディレクトリ
        output_dir = self.output_dir.get()  # 出力画像ディレクトリ
        n_symbols_per_card = self.n_symbols.get()  # カード1枚当たりのシンボル数
        n_voronoi_iters = 2 * self.cvt_level.get()  # 重心ボロノイ分割の反復回数
        # card_size_mm: カードの長辺サイズ[mm]
        # card_img_size: カード1枚当たりの画像サイズ (intなら円、(幅, 高さ) なら矩形で作成) [pix]
        if _card_shape == "円":
            card_size_mm = self.card_width.get()
            card_img_size = int(card_size_mm * pix_per_mm)
        else:
            card_size_mm = max(self.card_width.get(), self.card_height.get())
            card_img_size = tuple(int(x * pix_per_mm) for x in (self.card_width.get(), self.card_height.get()))
        card_margin = int(self.card_margin.get() * pix_per_mm)  # カード1枚の余白サイズ [pix]
        page_size_mm = (self.page_width.get(), self.page_height.get())  # PDFの(幅, 高さ)[mm]
        seed = int(self.seed.get())  # 乱数種

        # 出力ディレクトリの確認
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) != 0:
            yes = messagebox.askyesno("確認", "出力フォルダが空ではありません。同名のファイルは上書きされますが、よろしいですか？")
            if not yes:
                return

        # 出力ディレクトリ作成
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            messagebox.showerror("エラー", "出力フォルダの作成に失敗しました")
            return

        # 乱数初期化
        random.seed(seed)
        np.random.seed(seed)

        # 各カード毎の組み合わせを生成
        pairs, n_symbols = make_dobble_deck(n_symbols_per_card)
        # シンボルの組み合わせをcsvで保存
        np.savetxt(os.path.join(output_dir, "pairs.csv"), np.array(pairs), fmt="%d", delimiter=",")

        # image_dirからn_symbols数の画像を取得
        try:
            images, _ = load_images(image_dir, n_symbols, shuffle=self.shuffle.get())
        except ValueError:
            messagebox.showerror("エラー", f"入力画像フォルダに{n_symbols}個以上の画像ファイル (jpg または png) が存在するか確認してください")
            return

        card_images = []
        for i, image_indexes in enumerate(pairs):
            path = output_dir + os.sep + f"{i}.png"

            card_img = layout_images_randomly_wo_overlap(
                images,
                image_indexes,
                card_img_size,
                card_margin,
                draw_frame=True,
                method="voronoi",
                n_voronoi_iters=n_voronoi_iters,
            )
            cv2.imwrite(path, card_img)

            card_images.append(card_img)

        # 各画像をA4 300 DPIに配置しPDF化
        images_to_pdf(
            card_images,
            output_dir + os.sep + "card.pdf",
            dpi=DPI,
            card_long_side_mm=card_size_mm,
            width_mm=page_size_mm[0],
            height_mm=page_size_mm[1],
        )

        messagebox.showinfo("完了", f"{output_dir}にファイルが生成されました")


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()