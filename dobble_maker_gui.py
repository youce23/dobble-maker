import json
import os
import random
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
from typing import Any, Literal, cast

import numpy as np

from card_drawer.draw_card import (
    CARD_SHAPE,
    images_to_pdf,
    layout_images_randomly_wo_overlap,
    load_images,
    make_image_of_thumbnails_with_names,
)
from cv2_image_utils import imwrite_japanese
from deck_generator.generator import get_valid_params, make_dobble_deck
from dobble_maker import (
    load_image_list,
    save_card_list_to_csv,
    sort_images_by_image_list,
)


class Application(tk.Frame):
    DPI = 300  # 解像度
    MM_PER_INCH = 25.4
    PIX_PER_MM = DPI / MM_PER_INCH  # 1mmあたりのピクセル数

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
        dir_name = filedialog.askdirectory(
            initialdir=dir, mustexist=False
        )  # NOTE: Windows環境では mustexist=False が効かなかった
        if dir_name == "":  # キャンセル押下時
            return
        self.output_dir.set(dir_name)

    def _select_image_list_file(self):
        # *.xlsx or *.csv ファイル選択ダイアログ
        path = self.image_list_file_path.get()
        if os.path.isfile(path):
            init_dir = os.path.dirname(path)
            init_file = os.path.basename(path)
        else:
            init_dir = self.input_dir.get()
            init_file = None
        path = filedialog.askopenfilename(
            filetypes=[("画像リストファイル", "*.xlsx;*.csv")], initialdir=init_dir, initialfile=init_file
        )
        if path == "":  # キャンセル押下時
            return
        self.image_list_file_path.set(path)

    def _limit_min_size_rate(self):
        mx_r = self.max_size_rate.get()
        self.min_size_rate_entry.configure(to=mx_r - 0.01)

    def _limit_max_size_rate(self):
        mn_r = self.min_size_rate.get()
        self.max_size_rate_entry.configure(from_=mn_r + 0.01)

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

        # デッキのタイプ
        self.deck_type = tk.StringVar(value="Normal")
        deck_type_label = tk.Label(self.master, text="デッキ構成")
        deck_type_entry = ttk.Combobox(
            self.master,
            width=12,
            height=3,
            state="readonly",
            values=("Normal", "Twin Symbols", "Triple Cards"),
            textvariable=self.deck_type,
        )

        # カードあたりのシンボル数を入力するリスト
        self.n_symbols_per_card = tk.IntVar(value=5)
        n_symbols_per_card_label = tk.Label(self.master, text="カードあたりのシンボル数")
        n_symbols_per_card_entry = ttk.Combobox(
            self.master, width=6, height=5, state="readonly", textvariable=self.n_symbols_per_card
        )

        # デッキの構成カード数を入力するリスト
        # ("Triple Cards"の場合のみ有効化する)
        self.n_cards = tk.IntVar()
        n_cards_label = tk.Label(self.master, text="全カード数")
        n_cards_entry = ttk.Combobox(self.master, width=6, height=5, state="disabled", textvariable=self.n_cards)

        # デッキタイプの入力にしたがってカード当たりのシンボル数, デッキのカード数を変更させる
        def _update_n_symbols_per_card_and_n_cards(event) -> None:
            deck_type = self.deck_type.get()
            if deck_type == "Normal":
                n_symbols_per_card_entry["values"] = [str(i) for i in get_valid_params()]
                n_symbols_per_card_entry.current(3)  # 初期値を選択
                n_cards_entry["values"] = []
                n_cards_entry.configure(state="disabled")
            elif deck_type == "Twin Symbols":
                n_symbols_per_card_entry["values"] = [
                    str(i) for i in get_valid_params(n_shared_symbols=2, n_shared_cards=2)
                ]
                n_symbols_per_card_entry.current(3)
                n_cards_entry["values"] = []
                n_cards_entry.configure(state="disabled")
            elif deck_type == "Triple Cards":
                _params = get_valid_params(n_shared_symbols=1, n_shared_cards=3)
                _n_symbols = sorted({i for i, _ in cast(list[tuple[int, int]], _params)})
                n_symbols_per_card_entry["values"] = [str(i) for i in _n_symbols]
                n_symbols_per_card_entry.current(3)

                n_symbols = self.n_symbols_per_card.get()
                _params = get_valid_params(n_shared_symbols=1, n_shared_cards=3, n_symbols_per_card=n_symbols)
                _n_cards = sorted({i for i in cast(list[int], _params)})
                n_cards_entry["values"] = [str(i) for i in _n_cards]
                n_cards_entry.configure(state="readonly")
                n_cards_entry.current(0)

        def _update_n_cards(event) -> None:
            deck_type = self.deck_type.get()
            if deck_type == "Triple Cards":
                n_symbols = self.n_symbols_per_card.get()
                n_cards_values = get_valid_params(n_shared_symbols=1, n_shared_cards=3, n_symbols_per_card=n_symbols)
                n_cards_entry["values"] = [str(i) for i in n_cards_values]
                n_cards_entry.current(0)

        deck_type_entry.bind("<<ComboboxSelected>>", _update_n_symbols_per_card_and_n_cards)
        n_symbols_per_card_entry.bind("<<ComboboxSelected>>", _update_n_cards)
        _update_n_symbols_per_card_and_n_cards(None)  # 初期値を設定するために実行

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
        page_width_entry = tk.Spinbox(self.master, width=6, from_=1, to=1200, increment=1, textvariable=self.page_width)

        # PDF ページ高さ [mm]
        self.page_height = tk.IntVar(value=297)
        page_height_label = tk.Label(self.master, text="高さ[mm]")
        page_height_entry = tk.Spinbox(
            self.master, width=6, from_=1, to=1200, increment=1, textvariable=self.page_height
        )

        # レイアウト調整
        layout_params_label = tk.Label(self.master, text="レイアウト調整")

        # レイアウトの均一性 (重心ボロノイ分割の反復回数)
        self.cvt_level = tk.IntVar(value=5)
        cvt_level_label = tk.Label(self.master, text="レイアウト均一性 (1から20)")
        cvt_level_entry = tk.Spinbox(
            self.master, state="readonly", width=6, from_=1, to=20, increment=1, textvariable=self.cvt_level
        )
        cvt_level_desc = tk.Label(self.master, text="大きいほどシンボルサイズが均一かつ整列したレイアウトになります")
        # シンボルサイズ比の調整パラメータ (power diagramsにおける各シンボルの半径調整パラメータ)
        self.cvt_radius = tk.IntVar(value=5)
        cvt_radius_label = tk.Label(self.master, text="シンボルサイズ比調整 (0から20)")
        cvt_radius_entry = tk.Spinbox(
            self.master, state="readonly", width=6, from_=0, to=20, increment=1, textvariable=self.cvt_radius
        )
        cvt_radius_desc = tk.Label(self.master, text="大きいほどシンボルサイズに差が生じやすくなります")
        # シンボルサイズの最小・最大サイズ
        self.min_size_rate = tk.DoubleVar(value=0.0)
        self.max_size_rate = tk.DoubleVar(value=1.0)
        min_size_rate_label = tk.Label(self.master, text="シンボルの最小サイズ")
        max_size_rate_label = tk.Label(self.master, text="シンボルの最大サイズ")
        self.min_size_rate_entry = tk.Spinbox(
            self.master,
            state="readonly",
            width=6,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.min_size_rate,
            command=self._limit_max_size_rate,
        )  # 範囲を動的に変更するためにselfに持たせる
        self.max_size_rate_entry = tk.Spinbox(
            self.master,
            state="readonly",
            width=6,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.max_size_rate,
            command=self._limit_min_size_rate,
        )  # 範囲を動的に変更するためにselfに持たせる
        min_size_rate_desc = tk.Label(self.master, text="カードサイズ長辺に対するシンボルサイズ長辺の比")
        max_size_rate_desc = tk.Label(self.master, text="カードサイズ長辺に対するシンボルサイズ長辺の比")

        # seed
        self.seed = tk.IntVar(value=0)
        seed_label = tk.Label(self.master, text="乱数seed (整数)")
        seed_entry = tk.Entry(self.master, width=6, textvariable=self.seed)

        # 画像リスト作成 (ラベルフレーム内に配置しチェックボックスでON/OFF制御)
        check_thumb_frame = tk.LabelFrame(self.master, text="label frame")
        # * チェックボックス (ラベルフレームの制御)
        _i_state = True  # チェックボックスの初期値
        self.check_thumb_group: list[tuple[tk.Entry | tk.Button | tk.Spinbox, str]] = (
            []
        )  # チェックボックスで有効/無効を切り替える(要素, 有効時のstate)を入れる
        self.check_thumb = tk.BooleanVar(value=_i_state)
        check_thumb_entry = tk.Checkbutton(
            self.master, text="シンボル一覧を作成", variable=self.check_thumb, command=self._change_state_by_check_thumb
        )
        check_thumb_frame["labelwidget"] = check_thumb_entry
        # * 画像リストファイルパス (*.xlsx, *.csv)
        self.image_list_file_path = tk.StringVar(value=self.input_dir.get() + os.sep + "画像リスト.xlsx")
        image_list_file_label = tk.Label(check_thumb_frame, text="入力ファイル")
        image_list_file_entry = tk.Entry(
            check_thumb_frame,
            width=64,
            textvariable=self.image_list_file_path,
            state=tk.NORMAL if _i_state else tk.DISABLED,
        )
        image_list_file_button = tk.Button(
            check_thumb_frame,
            text="...",
            command=self._select_image_list_file,
            state=tk.NORMAL if _i_state else tk.DISABLED,
        )
        self.check_thumb_group.append((image_list_file_entry, tk.NORMAL))
        self.check_thumb_group.append((image_list_file_button, tk.NORMAL))
        # * テーブルサイズ
        self.table_rows = tk.IntVar(value=7)
        self.table_cols = tk.IntVar(value=6)
        table_rows_label = tk.Label(check_thumb_frame, text="行数")
        table_cols_label = tk.Label(check_thumb_frame, text="列数")
        table_rows_entry = tk.Spinbox(
            check_thumb_frame,
            state="readonly" if _i_state else tk.DISABLED,
            width=6,
            from_=1,
            to=30,
            increment=1,
            textvariable=self.table_rows,
        )
        table_cols_entry = tk.Spinbox(
            check_thumb_frame,
            state="readonly" if _i_state else tk.DISABLED,
            width=6,
            from_=1,
            to=30,
            increment=1,
            textvariable=self.table_cols,
        )
        self.check_thumb_group.append((table_rows_entry, "readonly"))
        self.check_thumb_group.append((table_cols_entry, "readonly"))
        # * 最大文字高さの割合
        self.text_h_rate = tk.DoubleVar(value=0.25)
        text_h_rate_label = tk.Label(check_thumb_frame, text="画像内の最大文字高さ比")
        text_h_rate_entry = tk.Spinbox(
            check_thumb_frame,
            state="readonly" if _i_state else tk.DISABLED,
            width=6,
            from_=0.01,
            to=0.5,
            increment=0.01,
            textvariable=self.text_h_rate,
        )
        self.check_thumb_group.append((text_h_rate_entry, "readonly"))
        # * 余白調整パラメータ
        self.thumb_margin_p = tk.DoubleVar(value=0.05)
        thumb_margin_p_label = tk.Label(check_thumb_frame, text="シンボルサイズ調整値")
        thumb_margin_p_entry = tk.Spinbox(
            check_thumb_frame,
            state="readonly" if _i_state else tk.DISABLED,
            width=6,
            from_=0,
            to=0.2,
            increment=0.005,
            textvariable=self.thumb_margin_p,
        )
        thumb_margin_p_desc = tk.Label(
            check_thumb_frame, text="値を大きくするとシンボルが小さくなり、カード(円)端に描画されやすくなります"
        )
        self.check_thumb_group.append((thumb_margin_p_entry, "readonly"))
        # * シンボル一覧のみ作成ボタン
        make_thumbs_button = tk.Button(
            check_thumb_frame,
            state=tk.NORMAL if _i_state else tk.DISABLED,
            text="シンボル一覧のみ作成",
            command=self._make_thumbnails,
        )
        self.check_thumb_group.append((make_thumbs_button, tk.NORMAL))

        # 計算実行ボタン
        calculate_button = tk.Button(self.master, text="計算実行", command=self._run)

        # ----------------
        # レイアウト (grid)
        # ----------------
        row = 0
        input_dir_label.grid(row=row, column=0, stick=tk.W)
        input_dir_entry.grid(row=row, column=1, stick=tk.W, columnspan=2)
        input_dir_button.grid(row=row, column=3, stick=tk.W)
        row += 1

        output_dir_label.grid(row=row, column=0, stick=tk.W)
        output_dir_entry.grid(row=row, column=1, stick=tk.W, columnspan=2)
        output_dir_button.grid(row=row, column=3, stick=tk.W)
        row += 1

        shuffle_label.grid(row=row, column=0, stick=tk.W)
        shuffle_entry.grid(row=row, column=1, sticky=tk.W)
        row += 1

        deck_type_label.grid(row=row, column=0, stick=tk.W)
        deck_type_entry.grid(row=row, column=1, stick=tk.W)
        row += 1

        n_symbols_per_card_label.grid(row=row, column=0, stick=tk.E)
        n_symbols_per_card_entry.grid(row=row, column=1, stick=tk.W)
        row += 1

        n_cards_label.grid(row=row, column=0, stick=tk.E)
        n_cards_entry.grid(row=row, column=1, stick=tk.W)
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

        layout_params_label.grid(row=row, column=0, stick=tk.W, columnspan=3)
        row += 1
        cvt_level_label.grid(row=row, column=0, stick=tk.E)
        cvt_level_entry.grid(row=row, column=1, stick=tk.W)
        cvt_level_desc.grid(row=row, column=2, stick=tk.W)
        row += 1
        cvt_radius_label.grid(row=row, column=0, stick=tk.E)
        cvt_radius_entry.grid(row=row, column=1, stick=tk.W)
        cvt_radius_desc.grid(row=row, column=2, stick=tk.W)
        row += 1
        min_size_rate_label.grid(row=row, column=0, stick=tk.E)
        self.min_size_rate_entry.grid(row=row, column=1, stick=tk.W)
        min_size_rate_desc.grid(row=row, column=2, stick=tk.W)
        row += 1
        max_size_rate_label.grid(row=row, column=0, stick=tk.E)
        self.max_size_rate_entry.grid(row=row, column=1, stick=tk.W)
        max_size_rate_desc.grid(row=row, column=2, stick=tk.W)
        row += 1

        seed_label.grid(row=row, column=0, stick=tk.W)
        seed_entry.grid(row=row, column=1, stick=tk.W)
        row += 1

        check_thumb_frame.grid(row=row, column=0, columnspan=4)
        # -- 以下はフレーム内に配置 ----
        fr_row = 0
        image_list_file_label.grid(row=fr_row, column=0, stick=tk.E)
        image_list_file_entry.grid(row=fr_row, column=1, stick=tk.W, columnspan=2)
        image_list_file_button.grid(row=fr_row, column=3, stick=tk.W)
        fr_row += 1
        table_rows_label.grid(row=fr_row, column=0, stick=tk.E)
        table_rows_entry.grid(row=fr_row, column=1, stick=tk.W)
        fr_row += 1
        table_cols_label.grid(row=fr_row, column=0, stick=tk.E)
        table_cols_entry.grid(row=fr_row, column=1, stick=tk.W)
        fr_row += 1
        text_h_rate_label.grid(row=fr_row, column=0, stick=tk.E)
        text_h_rate_entry.grid(row=fr_row, column=1, stick=tk.W)
        fr_row += 1
        thumb_margin_p_label.grid(row=fr_row, column=0, stick=tk.E)
        thumb_margin_p_entry.grid(row=fr_row, column=1, stick=tk.W)
        thumb_margin_p_desc.grid(row=fr_row, column=2, stick=tk.W, columnspan=2)
        fr_row += 1
        make_thumbs_button.grid(row=fr_row, column=0, pady=5, columnspan=4)
        fr_row += 1
        # -- ここまで --
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

    def _change_state_by_check_thumb(self):
        # 画像リスト作成チェックボックスのON/OFFで、関連する要素の有効/無効を切り替え
        for entry, state in self.check_thumb_group:
            entry.config(state=state if self.check_thumb.get() else tk.DISABLED)

    def _initialize(self) -> bool:
        err_msg = self._error_check_params()
        if err_msg != "":
            messagebox.showerror("エラー", err_msg)
            return False

        # 各パラメータを取得
        _card_shape: Literal["円", "長方形"] = self.card_shape.get()

        image_dir = self.input_dir.get()  # 入力画像ディレクトリ
        output_dir = self.output_dir.get()  # 出力画像ディレクトリ
        deck_type = self.deck_type.get()  # デッキ構成
        n_symbols_per_card = self.n_symbols_per_card.get()  # カード1枚当たりのシンボル数
        n_cards = self.n_cards.get()  # 総カード数
        n_voronoi_iters = 2 * self.cvt_level.get()  # 重心ボロノイ分割の反復回数
        radius_p = self.cvt_radius.get() / 10  # 重心ボロノイ分割の母点の半径調整パラメータ
        min_image_size_rate = (
            self.min_size_rate.get()
        )  # 最小シンボル画像サイズ (カード長辺に対するシンボル画像長辺の比)
        max_image_size_rate = (
            self.max_size_rate.get()
        )  # 最大シンボル画像サイズ (カード長辺に対するシンボル画像長辺の比)

        # card_size_mm: カードの長辺サイズ[mm]
        # card_img_size: カード1枚当たりの画像サイズ (intなら円、(幅, 高さ) なら矩形で作成) [pix]
        if _card_shape == "円":
            card_shape = CARD_SHAPE.CIRCLE
            card_size_mm = self.card_width.get()
            card_img_size = int(card_size_mm * Application.PIX_PER_MM)
        else:
            card_shape = CARD_SHAPE.RECTANGLE
            card_size_mm = max(self.card_width.get(), self.card_height.get())
            card_img_size = (
                int(self.card_width.get() * Application.PIX_PER_MM),
                int(self.card_height.get() * Application.PIX_PER_MM),
            )
        card_margin = int(self.card_margin.get() * Application.PIX_PER_MM)  # カード1枚の余白サイズ [pix]
        page_size_mm = (self.page_width.get(), self.page_height.get())  # PDFの(幅, 高さ)[mm]
        seed = int(self.seed.get())  # 乱数種

        # 出力ディレクトリの確認
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) != 0:
            yes = messagebox.askyesno(
                "確認", "出力フォルダが空ではありません。同名のファイルは上書きされますが、よろしいですか？"
            )
            if not yes:
                return False

        # 出力ディレクトリ作成
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            messagebox.showerror("エラー", "出力フォルダの作成に失敗しました")
            return False

        # 乱数初期化
        random.seed(seed)
        np.random.seed(seed)

        # パラメータ設定
        self._seed = seed
        self._deck_type = deck_type
        self._n_symbols_per_card = n_symbols_per_card
        self._n_cards = n_cards
        self._output_dir = output_dir
        self._image_dir = image_dir
        self._card_size_mm = card_size_mm
        self._card_img_size = card_img_size
        self._card_margin = card_margin
        self._card_shape = card_shape
        self._radius_p = radius_p
        self._n_voronoi_iters = n_voronoi_iters
        self._min_image_size_rate = min_image_size_rate
        self._max_image_size_rate = max_image_size_rate
        self._page_size_mm = page_size_mm

        if self.check_thumb.get():
            self._image_table_size = (self.table_rows.get(), self.table_cols.get())
            self._thumb_margin = self.thumb_margin_p.get()
            self._text_h_rate = self.text_h_rate.get()
            self._image_list_file_path = self.image_list_file_path.get()

        return True

    def _run(self):
        try:
            if not self._initialize():
                return

            # 各カード毎の組み合わせを生成
            match self._deck_type:
                case "Normal":
                    n_shared_symbols = 1
                    n_shared_cards = 2
                    n_cards = None
                case "Twin Symbols":
                    n_shared_symbols = 2
                    n_shared_cards = 2
                    n_cards = None
                case "Triple Cards":
                    n_shared_symbols = 1
                    n_shared_cards = 3
                    n_cards = self._n_cards
            pairs, n_symbols = make_dobble_deck(
                self._n_symbols_per_card,
                n_shared_symbols=n_shared_symbols,
                n_shared_cards=n_shared_cards,
                n_cards=n_cards,
            )
            save_card_list_to_csv(self._output_dir, pairs)  # 組み合わせのcsvを出力

            # image_dirからn_symbols数の画像を取得
            try:
                images, image_paths = load_images(self._image_dir, n_symbols, shuffle=self.shuffle.get())
            except ValueError:
                messagebox.showerror(
                    "エラー",
                    f"入力画像フォルダに{n_symbols}個以上の画像ファイル (jpg または png) が存在するか確認してください",
                )
                return

            card_images = []
            for i, image_indexes in enumerate(pairs):
                path = self._output_dir + os.sep + f"{i}.png"

                card_img = layout_images_randomly_wo_overlap(
                    images,
                    image_indexes,
                    self._card_img_size,
                    self._card_margin,
                    self._card_shape,
                    draw_frame=True,
                    method="voronoi",
                    radius_p=self._radius_p,
                    n_voronoi_iters=self._n_voronoi_iters,
                    min_image_size_rate=self._min_image_size_rate,
                    max_image_size_rate=self._max_image_size_rate,
                )
                imwrite_japanese(path, card_img)

                card_images.append(card_img)

            # シンボル一覧画像の作成
            image_names = None
            if self.check_thumb.get():
                thumb_card_images, image_names = self._make_thumbnails_core(images, image_paths)
                card_images.extend(thumb_card_images)

            # カード、シンボル画像に関する情報をcsv出力
            save_card_list_to_csv(self._output_dir, pairs, image_paths=image_paths, image_names=image_names)

            # 各画像をA4 300 DPIに配置しPDF化
            images_to_pdf(
                card_images,
                self._output_dir + os.sep + "card.pdf",
                dpi=Application.DPI,
                card_long_side_mm=self._card_size_mm,
                width_mm=self._page_size_mm[0],
                height_mm=self._page_size_mm[1],
            )

            # 生成パラメータを保存
            self._save_params()

            messagebox.showinfo("完了", f"{self._output_dir}にファイルが生成されました")
        except Exception as e:
            messagebox.showerror("エラー", str(e))

    def _save_params(self, *, save_card_params: bool = True, save_thumb_params: bool = True) -> None:
        params: dict[str, Any] = {}

        # 既存のパラメータファイルを読み込み
        json_path = os.path.join(self._output_dir, "parameters.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, mode="r", encoding="utf_8") as f:
                    params = json.load(f)
            except Exception:
                messagebox.showwarning(
                    "警告",
                    f"{json_path}の読み込みに失敗したため、同ファイルは更新されません",
                )
                return

        if save_card_params:
            params |= {
                "image_dir": self._image_dir,
                "output_dir": self._output_dir,
                "shuffle": self.shuffle.get(),
                "deck_type": self._deck_type,
                "n_symbols_per_card": self._n_symbols_per_card,
            }
            if self._deck_type == "Triple Cards":
                params |= {"n_cards": self._n_cards}
            params |= {
                "card_shape": self._card_shape,
                "card_width_mm": self.card_width.get(),
                "card_height_mm": self.card_height.get(),
                "card_margin_mm": self.card_margin.get(),
                "page_size_mm": self._page_size_mm,
                "layout_uniform_level": self.cvt_level.get(),
                "symbol_size_ratio_level": self.cvt_radius.get(),
                "min_image_size_rate": self._min_image_size_rate,
                "max_image_size_rate": self._max_image_size_rate,
                "seed": self._seed,
            }
        if save_thumb_params and self.check_thumb.get():
            params["thumbnails"] = {
                "image_list_file_path": self._image_list_file_path,
                "image_table_size": self._image_table_size,
                "text_h_rate": self._text_h_rate,
                "symbol_size_p": self._thumb_margin,
            }
        try:
            with open(json_path, mode="w", encoding="utf_8") as f:
                json.dump(params, f, ensure_ascii=False, indent=4)
        except Exception:
            messagebox.showwarning(
                "警告",
                f"{json_path}の保存に失敗したため、同ファイルは更新されません",
            )
            return

    def _make_thumbnails_core(
        self, images: list[np.ndarray], image_paths: list[str]
    ) -> tuple[list[np.ndarray], dict[str, str]]:
        card_images: list[np.ndarray] = []

        image_names = load_image_list(self._image_list_file_path)
        sorted_images, sorted_names = sort_images_by_image_list(
            images, image_paths, image_names
        )  # image_namesの順序でimage_pathsをソート
        thumbs_cards = make_image_of_thumbnails_with_names(
            self._card_shape,
            self._card_img_size,
            self._card_margin,
            self._image_table_size,
            sorted_images,
            sorted_names,
            thumb_margin=self._thumb_margin,
            text_h_rate=self._text_h_rate,
            draw_frame=True,
        )  # 画像をサムネイル化したカード画像を作成
        for i, card in enumerate(thumbs_cards):
            path = self._output_dir + os.sep + f"thumbnail_{i}.png"
            imwrite_japanese(path, card)
            card_images.append(card)

        return card_images, image_names

    def _make_thumbnails(self):
        if not self._initialize():
            return

        # 各カード毎の組み合わせを生成
        _, n_symbols = make_dobble_deck(self._n_symbols_per_card)

        # image_dirからn_symbols数の画像を取得
        try:
            images, image_paths = load_images(self._image_dir, n_symbols, shuffle=self.shuffle.get())
        except ValueError:
            messagebox.showerror(
                "エラー",
                f"入力画像フォルダに{n_symbols}個以上の画像ファイル (jpg または png) が存在するか確認してください",
            )
            return

        # カード作成
        try:
            _ = self._make_thumbnails_core(images, image_paths)
        except Exception as e:
            messagebox.showerror("エラー", str(e))
            return

        # 生成パラメータを保存
        self._save_params(save_card_params=False, save_thumb_params=True)

        messagebox.showinfo("完了", f"{self._output_dir}にシンボル一覧画像が生成されました")

        return


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
