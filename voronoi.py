from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon


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
    bnd_pts: list[tuple[float, float]], sites: list[tuple[float, float]], *, show: bool = False
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
        show (bool, optional):
            Trueならボロノイ図を描画. Defaults to False.

    Returns:
        list[list[tuple[float, float]]]: 各ボロノイ領域の境界座標セット
    """
    # すべての母点のボロノイ領域を有界にするために，ダミー母点(境界の外側であること)を追加
    dummy_sites = np.array([[1e6, 1e6], [1e6, -1e6], [-1e6, 0]])
    gn_sites = np.concatenate([sites, dummy_sites])

    # ボロノイ図の計算
    vor = Voronoi(gn_sites)

    # 分割する領域をPolygonに
    bnd_poly = Polygon(bnd_pts)

    # 各ボロノイ領域の頂点座標を格納するリスト
    vor_polys: list[list[tuple[float, float]]] = []

    # ダミー以外の母点についての繰り返し
    for i in range(len(gn_sites) - dummy_sites.shape[0]):
        # 閉空間を考慮しないボロノイ領域
        vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        # 分割する領域をボロノイ領域の共通部分を計算
        i_cell = bnd_poly.intersection(Polygon(vor_poly))

        # 閉空間を考慮したボロノイ領域の頂点座標を格納
        vor_polys.append(list(i_cell.exterior.coords[:-1]))

    if show:
        # ボロノイ図の描画
        fig = plt.figure(figsize=(7, 6), num=1)
        ax = fig.add_subplot(111)

        # 母点
        pnts_np = np.array(sites)
        ax.scatter(pnts_np[:, 0], pnts_np[:, 1])

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

        plt.show()

    assert len(vor_polys) == len(sites)

    return vor_polys


def cvt(
    bounding_points: list[tuple[float, float]], n_site: int, *, n_iters: int = 10, show_step: range | None = None
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    """重心ボロノイ分割(Centroidal Voronoi Tessellation)

    Lloyd's relaxationでCVTする

    Args:
        bounding_points (list[tuple[float, float]]):
            ボロノイ分割をする範囲を定義する凸包の境界
        n_site (int): 母点(site)の数
        n_iters (int, optional): 重心を得るための反復数. Defaults to 10.
        show_step (range | None): 指定された回数の時にボロノイ図を表示する (Noneなら常に非表示)

    Returns:
        list[tuple[float, float]]: 各ボロノイ領域の重心
        list[list[tuple[float, float]]]]: 各ボロノイ領域の境界座標セット
    """
    # 母点の初期値として、領域に含まれるランダムな点群を生成
    points = cast(list[tuple[float, float]], _rand_pts_in_poly(bounding_points, n_site))

    for i in range(n_iters):
        # 有限ボロノイ図を計算
        show = (show_step is not None) and (i in show_step)
        vor_regions = _bounded_voronoi(bounding_points, points, show=show)

        # 新しい点(重心)を計算
        new_points: list[tuple[float, float]] = []
        for reg in vor_regions:
            poly = Polygon(reg)
            new_points.append((poly.centroid.x, poly.centroid.y))

        points = new_points

    return points, vor_regions


if __name__ == "__main__":
    # 動作確認
    n_sites = 12  # 母点
    n_iters = 20  # 反復数

    # 円でボロノイ分割
    circle = [(np.cos(x / 180.0 * np.pi), np.sin(x / 180.0 * np.pi)) for x in range(0, 360, 5)]
    cvt(circle, n_sites, n_iters=n_iters, show_step=range(0, n_iters, n_iters - 1))

    # 矩形でボロノイ分割
    box = [(0, 0), (1, 0), (1, 1.5), (0, 1.5)]
    cvt(box, n_sites, n_iters=n_iters, show_step=range(0, n_iters, n_iters - 1))
