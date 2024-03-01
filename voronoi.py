from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon

from laguerre_voronoi_2d import convert_power_diagrams_to_cells, power_diagrams


def _rand_pts_in_poly(
    bounding_points: list[tuple[float, float]], n_pts: int, *, as_point: bool = False
) -> list[Point | tuple[float, float]]:
    """
    任意の凸包内に含まれるランダム点群を生成
    (seedを設定する場合はnp.random.seedを使用する)

    参考
    * Generate Random Points in a Polygon
      * https://www.matecdev.com/posts/random-points-in-polygon.html
      * 以下はslowな方法であり、Geopandasを使ったfastな方法も紹介されている

    Args:
        bounding_points (list[tuple[float, float]]): 凸包の頂点群
        n_pts (int): 生成する点群数
        as_point: 戻り値の型を指定

    Returns:
        list[tuple[float, float] | Point]:
            as_pointがTrueならPoint, Falseならtuple[float, float]でランダム点群を返す
    """
    # 凸包を生成
    bounding_poly = Polygon(bounding_points)

    points: list[Point | tuple[float, float]] = []
    x0, y0, x1, y1 = bounding_poly.bounds  # いったん矩形範囲を取得

    # ランダム点群を1つずつ生成
    while len(points) < n_pts:
        pnt = Point(np.random.uniform(x0, x1), np.random.uniform(y0, y1))
        if bounding_poly.contains(pnt):
            if as_point:
                points.append(pnt)
            else:
                points.append((pnt.x, pnt.y))

    return points


def _bounded_voronoi(
    bnd_pts: list[tuple[float, float]],
    sites: list[tuple[float, float]],
    *,
    radius: np.ndarray | None = None,
    show: bool = False,
) -> list[list[tuple[float, float]]]:
    """
    有界なボロノイ図を計算

    参考:
    * "Pythonで有界な（閉じた）ボロノイ図を計算・描画する"
      * https://qiita.com/supbon2/items/30e0cb49c9338e721b8c

    Args:
        bnd_pts (list[tuple[float, float]]):
            ボロノイ分割をする範囲を定義する凸包の境界
        sites (list[tuple[float, float]]):
            母点
        radius:
            各母点の半径(Noneならボロノイ分割, 指定時はpower diagramsとして処理)
        show (bool, optional):
            Trueならボロノイ図を描画. Defaults to False.

    Returns:
        list[list[tuple[float, float]]]: 各ボロノイ領域の境界座標セット
    """
    # すべての母点のボロノイ領域を有界にするために，ダミー母点を追加
    # (ダミー母点が成す多角形にsitesが内包されること)
    _poly = Polygon(bnd_pts)
    _x0, _y0, _x1, _y1 = _poly.bounds
    _center = np.array((_poly.centroid.x, _poly.centroid.y))
    dummy_sites = (np.array([(_x0, _y0), (_x0, _y1), (_x1, _y1), (_x1, _y0)]) - _center) * 2 + _center
    gn_sites = np.concatenate([sites, dummy_sites])

    # ボロノイ図の計算
    if radius is None:
        vor = Voronoi(gn_sites)
    else:
        # power diagramsを計算すると、母点の位置と radius の関係によっては例外が発生することがある
        # その場合はVoronoiで計算する (母点が近接していると初期の iteration で発生しやすい)
        try:
            gn_radius = np.concatenate([radius, np.zeros(len(dummy_sites))])
            a, _ = power_diagrams(np.array(gn_sites), gn_radius)
            if len(a) < len(gn_sites):
                raise ValueError("領域を持たない母点が存在")
            power_poly = convert_power_diagrams_to_cells(a)
            assert sorted(power_poly.keys()) == list(range(len(gn_sites)))  # データが0からすべてそろっている
        except Exception:
            radius = None
            vor = Voronoi(gn_sites)

    # 分割する領域をPolygonに
    bnd_poly = Polygon(bnd_pts)

    # 各ボロノイ領域の頂点座標を格納するリスト
    vor_polys: list[list[tuple[float, float]]] = []

    # ダミー以外の母点についての繰り返し
    for i in range(len(gn_sites) - dummy_sites.shape[0]):
        # 閉空間を考慮しないボロノイ領域
        if radius is None:
            vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
            vor_poly = Polygon(vor_poly)
        else:
            vor_poly = Polygon(power_poly[i])
            if not vor_poly.is_simple:
                # ポリゴンが捻じれていたら直す
                # NOTE: 対処療法。なぜ捻じれるのかよくわからない。
                vor_poly = vor_poly.convex_hull
        # 分割する領域をボロノイ領域の共通部分を計算
        i_cell = bnd_poly.intersection(vor_poly)

        # 閉空間を考慮したボロノイ領域の頂点座標を格納
        vor_polys.append(list(i_cell.exterior.coords[:-1]))

    if show:
        # ボロノイ図の描画
        fig = plt.figure(figsize=(7, 6), num=1)
        fig.clear()
        ax = fig.add_subplot(111)

        # 母点
        pnts_np = np.array(sites)
        ax.scatter(pnts_np[:, 0], pnts_np[:, 1])

        # 母点の半径
        if radius is not None:
            for s, r in zip(sites, radius):
                ax.add_artist(plt.Circle(s, r, fill=True, alpha=0.4, lw=0.0, color="#8080f0", zorder=1))

        # ボロノイ領域
        poly_vor = PolyCollection(vor_polys, edgecolor="black", facecolors="None", linewidth=1.0)
        ax.add_collection(poly_vor)

        bnd_np = np.array(bnd_pts)
        xmin = np.min(bnd_np[:, 0])
        xmax = np.max(bnd_np[:, 0])
        ymin = np.min(bnd_np[:, 1])
        ymax = np.max(bnd_np[:, 1])

        ax.set_xlim(xmin - 0.1, xmax + 0.1)
        ax.set_ylim(ymin - 0.1, ymax + 0.1)
        ax.set_aspect("equal")

        # plt.show()
        plt.pause(0.01)

    assert len(vor_polys) == len(sites)

    return vor_polys


def cvt(
    bounding_points: list[tuple[float, float]],
    n_sites: int,
    *,
    radius_p: float = 0.0,
    n_iters: int = 10,
    show_step: range | None = None,
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    """重心ボロノイ分割(Centroidal Voronoi Tessellation)

    Lloyd's relaxationでCVTする

    Args:
        bounding_points (list[tuple[float, float]]):
            ボロノイ分割をする範囲を定義する凸包の境界
        n_sites (int): 母点(site)の数
        radius_p:
            0.0: ボロノイ分割
            >0.0: power diagamsにおける各母点の半径を調整するパラメータ
        n_iters (int, optional): 重心を得るための反復数. Defaults to 10.
        show_step (range | None): 指定された回数の時にボロノイ図を表示する (Noneなら常に非表示)

    Returns:
        list[tuple[float, float]]: 各ボロノイ領域の重心
        list[list[tuple[float, float]]]]: 各ボロノイ領域の境界座標セット
    """
    if n_iters < 1:
        raise ValueError(f"n_iters ({n_iters}) は 1 以上でなければならない")

    # 母点の初期値として、領域に含まれるランダムな点群を生成
    points = cast(list[tuple[float, float]], _rand_pts_in_poly(bounding_points, n_sites))
    radius = None
    if radius_p > 0.0:
        poly = Polygon(bounding_points)
        area_per_site = poly.area / n_sites  # 外側の図形の面積を母点で均等に分割した場合の面積を radius の基準とする
        r_basis = np.sqrt(area_per_site / np.pi)  # 基準半径
        radius = r_basis / 2 + radius_p * r_basis / 2 * np.random.random(n_sites)

    for i in range(n_iters):
        # 有限ボロノイ図を計算
        show = (show_step is not None) and (i in show_step)
        vor_regions = _bounded_voronoi(bounding_points, points, radius=radius, show=show)

        # 新しい点(重心)を計算
        new_points: list[tuple[float, float]] = []
        for reg in vor_regions:
            poly = Polygon(reg)
            new_points.append((poly.centroid.x, poly.centroid.y))

        points = new_points

    return points, vor_regions


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    # 動作確認
    n_sites = 12  # 母点
    n_iters = 20  # 反復数
    cpd_rate = 2  # CPDの領域間のサイズ比調整値, 0 以上 おおむね 2 以下 (0だとほぼvoronoiと同じ, 大きすぎても内部でvoronoiと同様に処理される)

    # 円でCPD
    circle = [(10000 * np.cos(x / 180.0 * np.pi), 5000 * np.sin(x / 180.0 * np.pi)) for x in range(0, 360, 5)]

    repeat = True
    while repeat:
        # 初期値によって例外が生じることがあるので、その場合はやり直す
        # (何回やっても通らない場合、cpd_rate (小さく), n_sites (減らす), 図形のアスペクト比(1:1に近づける)を見直し)
        repeat = False
        try:
            cvt(circle, n_sites, radius_p=cpd_rate, n_iters=n_iters, show_step=range(n_iters))
        except Exception:
            repeat = True

    # 矩形でCPD
    box = [(0, 0), (1, 0), (1, 1.5), (0, 1.5)]

    repeat = True
    while repeat:
        repeat = False
        try:
            cvt(box, n_sites, radius_p=cpd_rate, n_iters=n_iters, show_step=range(n_iters))
        except Exception:
            repeat = True

    # 円でボロノイ分割
    circle = [(np.cos(x / 180.0 * np.pi), np.sin(x / 180.0 * np.pi)) for x in range(0, 360, 5)]
    cvt(circle, n_sites, n_iters=n_iters, show_step=range(n_iters))

    # 矩形でボロノイ分割
    box = [(0, 0), (1, 0), (1, 1.5), (0, 1.5)]
    cvt(box, n_sites, n_iters=n_iters, show_step=range(n_iters))
