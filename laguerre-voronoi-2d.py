"""
https://gist.github.com/marmakoide/45d5389252683ae09c2df49d0548a627

MIT License

Copyright (c) 2021 Devert Alexandre

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

import itertools
from typing import Literal, cast

import numpy as np
from matplotlib import pyplot as plot
from matplotlib.collections import LineCollection
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import ConvexHull

# --- Misc. geometry code -----------------------------------------------------

"""
Pick N points uniformly from the unit disc
This sampling algorithm does not use rejection sampling.
"""


def disc_uniform_pick(N: int) -> NDArray:
    """単位円内のランダムな点を取得

    Args:
        N (int): 点数

    Returns:
        np.ndarray (N, 2): 単位円内の(x, y)座標
    """
    angle: ArrayLike = (2 * np.pi) * np.random.random(N)  # N個のランダムな角度
    out: NDArray = np.stack([np.cos(angle), np.sin(angle)], axis=1)  # angle に対応する単位円上の(x, y)
    out *= np.sqrt(np.random.random(N))[:, None]  # 単位円状の点をランダムに単位円内に配置

    assert out.shape == (N, 2)
    return out


def norm2(X: ArrayLike) -> np.float64:
    """ベクトル X の L2 ノルム"""
    return np.sqrt(np.sum(X**2))


def normalized(X: ArrayLike) -> ArrayLike:
    """ベクトル X を単位ベクトル化"""
    return X / norm2(X)


# --- Delaunay triangulation --------------------------------------------------


def get_triangle_normal(A: ArrayLike, B: ArrayLike, C: ArrayLike) -> ArrayLike:
    assert len(A) == len(B) == len(C) == 3

    v = normalized(np.cross(A, B) + np.cross(B, C) + np.cross(C, A))
    assert len(v) == 3
    return v


def get_power_circumcenter(A: ArrayLike, B: ArrayLike, C: ArrayLike) -> ArrayLike:
    assert len(A) == len(B) == len(C) == 3

    N = get_triangle_normal(A, B, C)
    v: ArrayLike = (-0.5 / N[2]) * N[:2]
    assert len(v) == 2
    return v


def is_ccw_triangle(A: ArrayLike, B: ArrayLike, C: ArrayLike) -> bool:
    assert len(A) == len(B) == len(C) == 2

    M = np.concatenate([np.stack([A, B, C]), np.ones((3, 1))], axis=1)
    return np.linalg.det(M) > 0


def get_power_triangulation(S: NDArray, R: ArrayLike) -> tuple[tuple[tuple[int, int, int]], np.ndarray]:
    assert S.shape == (len(R), 2)

    # Compute the lifted weighted points
    S_norm: ArrayLike = np.sum(S**2, axis=1) - R**2
    S_lifted: NDArray = np.concatenate([S, S_norm[:, None]], axis=1)

    assert len(S_norm) == len(R)
    assert S_lifted.shape == (len(R), 3)

    # Special case for 3 points
    if S.shape[0] == 3:
        if is_ccw_triangle(S[0], S[1], S[2]):
            return [[0, 1, 2]], np.array([get_power_circumcenter(*S_lifted)])
        else:
            return [[0, 2, 1]], np.array([get_power_circumcenter(*S_lifted)])

    # Compute the convex hull of the lifted weighted points
    hull = ConvexHull(S_lifted)

    # Extract the Delaunay triangulation from the lower hull
    tri_list: tuple[tuple[int, int, int]] = tuple(
        (a, b, c) if is_ccw_triangle(S[a], S[b], S[c]) else (a, c, b)
        for (a, b, c), eq in zip(hull.simplices, hull.equations)
        if eq[2] <= 0
    )

    # Compute the Voronoi points
    V: NDArray = np.array([get_power_circumcenter(*S_lifted[list(tri)]) for tri in tri_list])

    # Job done
    assert V.shape == (len(tri_list), 2)
    return tri_list, V


# --- Compute Voronoi cells ---------------------------------------------------

"""
Compute the segments and half-lines that delimits each Voronoi cell
* The segments are oriented so that they are in CCW order
* Each cell is a list of (i, j), (A, U, tmin, tmax) where
    * i, j are the indices of two ends of the segment. Segments end points are
    the circumcenters. If i or j is set to None, then it's an infinite end
    * A is the origin of the segment
    * U is the direction of the segment, as a unit vector
    * tmin is the parameter for the left end of the segment. Can be -1, for minus infinity
    * tmax is the parameter for the right end of the segment. Can be -1, for infinity
    * Therefore, the endpoints are [A + tmin * U, A + tmax * U]

各ボロノイセルを区切るセグメントとハーフラインを計算します。
* セグメントはCCW（反時計回り）の順序であるように向き付けられています。
* 各セルは (i, j), (A, U, tmin, tmax) のリストで構成されます。ここで
    * i, j はセグメントの2つの端点のインデックスです。セグメントの端点は外接円の中心です。
    i または j が None に設定されている場合、それは無限の端点です。
    * A はセグメントの原点です。
    * U はセグメントの方向で、単位ベクトルとして表現されます。
    * tmin はセグメントの左端のパラメータです。マイナス無限大の場合、-1 になります。
    * tmax はセグメントの右端のパラメータです。無限大の場合、-1 になります。
    * したがって、端点は [A + tmin * U, A + tmax * U] です。
"""

VOR_DATA_TYPE = list[tuple[tuple[int, int], tuple[ArrayLike, ArrayLike, Literal[0] | None, float | Literal[0] | None]]]


def get_voronoi_cells(S: NDArray, V: NDArray, tri_list: tuple[tuple[int, int, int]]) -> dict[int, VOR_DATA_TYPE]:
    assert S.shape[1] == 2
    assert V.shape == (len(tri_list), 2)

    # Keep track of which circles are included in the triangulation
    vertices_set = frozenset(itertools.chain(*tri_list))
    assert len(vertices_set) <= S.shape[0]

    # Keep track of which edge separate which triangles
    edge_map: dict[tuple[int, int], list[int]] = {}
    for i, tri in enumerate(tri_list):
        for edge in itertools.combinations(tri, 2):
            edge = cast(tuple[int, int], tuple(sorted(edge)))
            if edge in edge_map:
                edge_map[edge].append(i)
            else:
                edge_map[edge] = [i]

    # For each triangle
    voronoi_cell_map: dict[int, VOR_DATA_TYPE] = {i: [] for i in vertices_set}

    for i, (a, b, c) in enumerate(tri_list):
        # For each edge of the triangle
        for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
            # Finite Voronoi edge
            edge = cast(tuple[int, int], tuple(sorted((u, v))))
            if len(edge_map[edge]) == 2:
                j, k = edge_map[edge]
                if k == i:
                    j, k = k, j

                # Compute the segment parameters
                U: ArrayLike = V[k] - V[j]
                assert len(U) == 2
                U_norm = norm2(U)

                # Add the segment
                voronoi_cell_map[u].append(((j, k), (V[j], U / U_norm, 0, U_norm)))
            else:
                # Infinite Voronoi edge
                # Compute the segment parameters
                A: ArrayLike = S[u]
                B: ArrayLike = S[v]
                C: ArrayLike = S[w]
                D: ArrayLike = V[i]
                assert len(A) == len(B) == len(C) == len(D) == 2
                U = normalized(B - A)
                I_: ArrayLike = A + np.dot(D - A, U) * U
                W = normalized(I_ - D)
                assert len(U) == len(I_) == len(W) == 2
                if np.dot(W, I_ - C) < 0:
                    W = -W

                # Add the segment
                voronoi_cell_map[u].append(((edge_map[edge][0], -1), (D, W, 0, None)))
                voronoi_cell_map[v].append(((-1, edge_map[edge][0]), (D, -W, None, 0)))

    # Order the segments
    def order_segment_list(segment_list: VOR_DATA_TYPE) -> VOR_DATA_TYPE:
        # Pick the first element
        first = min((seg[0][0], i) for i, seg in enumerate(segment_list))[1]

        # In-place ordering
        segment_list[0], segment_list[first] = segment_list[first], segment_list[0]
        for i in range(len(segment_list) - 1):
            for j in range(i + 1, len(segment_list)):
                if segment_list[i][0][1] == segment_list[j][0][0]:
                    segment_list[i + 1], segment_list[j] = segment_list[j], segment_list[i + 1]
                    break

        # Job done
        return segment_list

    # Job done
    return {i: order_segment_list(segment_list) for i, segment_list in voronoi_cell_map.items()}


# --- Plot all the things -----------------------------------------------------


def display(S, R, tri_list, voronoi_cell_map):
    # Setup
    fig, ax = plot.subplots()
    plot.axis("equal")
    plot.axis("off")

    # Set min/max display size, as Matplotlib does it wrong
    min_corner = np.amin(S, axis=0) - np.max(R)
    max_corner = np.amax(S, axis=0) + np.max(R)
    plot.xlim((min_corner[0], max_corner[0]))
    plot.ylim((min_corner[1], max_corner[1]))

    # Plot the samples
    for Si, Ri in zip(S, R):
        ax.add_artist(plot.Circle(Si, Ri, fill=True, alpha=0.4, lw=0.0, color="#8080f0", zorder=1))

    # Plot the power triangulation
    edge_set = frozenset(tuple(sorted(edge)) for tri in tri_list for edge in itertools.combinations(tri, 2))
    line_list = LineCollection([(S[i], S[j]) for i, j in edge_set], lw=1.0, colors=".9")
    line_list.set_zorder(0)
    ax.add_collection(line_list)

    # Plot the Voronoi cells
    edge_map = {}
    for segment_list in voronoi_cell_map.values():
        for edge, (A, U, tmin, tmax) in segment_list:
            edge = tuple(sorted(edge))
            if edge not in edge_map:
                if tmax is None:
                    tmax = 10
                if tmin is None:
                    tmin = -10

                edge_map[edge] = (A + tmin * U, A + tmax * U)

    line_list = LineCollection(edge_map.values(), lw=1.0, colors="k")
    line_list.set_zorder(0)
    ax.add_collection(line_list)

    # Job done
    plot.show()


# --- power diagrams --------------------------------------------------------
def power_diagrams(sites: NDArray, radius: ArrayLike):
    """power diagramsを計算

    Args:
        sites (NDArray): 母点
        radius (ArrayLike): 母点ごとの半径, 全て0だとボロノイ分割と同じ
    """
    n_sites = sites.shape[0]
    assert sites.shape[1] == 2
    assert len(radius) == n_sites

    # Compute the power triangulation of the circles
    tri_list, V = get_power_triangulation(sites, radius)

    assert V.shape == (len(tri_list), 2)

    # Compute the Voronoi cells
    voronoi_cell_map = get_voronoi_cells(sites, V, tri_list)
    assert len(voronoi_cell_map) <= n_sites  # 各母点の強度次第で領域を持たない母点が生じることがある

    return voronoi_cell_map, tri_list


def convert_power_diagrams_to_cells(voronoi_cell_map: dict[int, VOR_DATA_TYPE]) -> dict[int, list[tuple[float, float]]]:
    """power_diagrams の戻り値からボロノイ領域毎の頂点座標に変換

    Args:
        voronoi_cell_map: power_diagrams()の同名の戻り値

    Returns:
        dict[int, list[tuple[float, float]]]: 頂点座標群
    """

    INF = 1e6

    vecs: dict[int, list[tuple[float, float]]] = {}
    for i, segment_list in voronoi_cell_map.items():
        edge_map = {}
        vecs[i] = []
        for edge, (A, U, tmin, tmax) in segment_list:
            edge = tuple(sorted(edge))
            if edge not in edge_map:
                if tmax is None:
                    tmax = INF
                if tmin is None:
                    tmin = -INF

                pt0 = A + tmin * U  # 線分の端点1
                pt1 = A + tmax * U  # 線分の端点2
                edge_map[edge] = (pt0, pt1)

                if not any([np.allclose(pt0, v) for v in vecs[i]]):
                    vecs[i].append(tuple(pt0))
                if not any([np.allclose(pt1, v) for v in vecs[i]]):
                    vecs[i].append(tuple(pt1))

    return vecs


# --- Main entry point --------------------------------------------------------


def main():
    # Generate samples, S contains circles center, R contains circles radius
    sample_count = 32
    while True:
        S: NDArray = 5 * disc_uniform_pick(sample_count)  # 指定半径の円内に sample_count の母点をサンプリング
        R: ArrayLike = 0.8 * np.random.random(sample_count) + 0.2  # 各母点の強度(円の半径)を設定

        assert S.shape == (sample_count, 2)
        assert len(R) == sample_count

        voronoi_cell_map, tri_list = power_diagrams(S, R)

        if len(voronoi_cell_map) == sample_count:
            # 全ての母点が領域を持つような初期値にならなかったらやり直す
            break

    # Display the result
    display(S, R, tri_list, voronoi_cell_map)

    return


if __name__ == "__main__":
    main()
