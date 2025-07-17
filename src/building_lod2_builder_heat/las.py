import math
import sys
from pathlib import Path

import laspy
import numpy as np
from laspy import LasData
from numpy.typing import NDArray
from pyproj import CRS
from shapely import Point

from building_lod2_builder_heat.bounds import GeoBounds
from building_lod2_builder_heat.outline import GeoOutline
from building_lod2_builder_heat.parameter import load_las_crs_from_json


def load_las(
    las_file_path: Path,
    canvas_size: tuple[int, int],
    max_factor: float = 4.0,
    outline: GeoOutline | None = None,
    canvas_bounds: GeoBounds | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], GeoBounds] | None:
    """
    DSMから画像と深度を読み込む。

    :param las_file_path: 読み込むDSMのファイルパス。
    :param canvas_size: 出力画像を収める最大サイズ。
    :param max_factor: 拡大時の最大倍率。
    :param outline: 拡大する際に対象物の輪郭を保つための補助とする対象物の外形線。
    :param canvas_bounds: 出力に収める座標範囲。
    :return: DSMから抽出したカラー画像と深度画像と座標範囲。
    """
    min_z_range = 10

    las_data: LasData
    with laspy.open(las_file_path) as las:
        las_data = las.read()
    crs: CRS | None = las_data.header.parse_crs()
    if crs is None:
        crs = load_las_crs_from_json(las_file_path.with_suffix(".json"))
    if crs is None:
        print(f"DSMの座標系を特定できませんでした", file=sys.stderr)
        return None

    if outline is not None:
        outline = outline.transform_to(crs)
    if canvas_bounds is not None:
        canvas_bounds = canvas_bounds.transform_to(crs)

    las_points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()
    las_grid_indices, las_x_values, las_y_values = _points_to_grid(las_points)
    las_width = len(las_x_values)
    las_height = len(las_y_values)

    las_x_min = float(las_data.header.x_min)
    las_x_max = float(las_data.header.x_max)
    las_y_min = float(las_data.header.y_min)
    las_y_max = float(las_data.header.y_max)
    las_bounds = GeoBounds(las_x_min, las_y_max, las_x_max, las_y_min, crs)
    las_x_step = las_bounds.width / las_width
    las_y_step = las_bounds.height / las_height

    # 実際の点の数の何倍の格子点があるか
    grid_density = las_width * las_height / las_data.header.point_count
    search_range = int(math.ceil(grid_density))

    def find_near_las_indices(
        point: tuple[float, float], i_range: range, j_range: range
    ) -> list[tuple[int, float]]:
        result = []
        for j in j_range:
            for i in i_range:
                if i < 0 or i >= las_width or j < 0 or j >= las_height:
                    continue
                idx = int(las_grid_indices[j, i])
                if idx < 0:
                    continue
                las_point = las_points[idx]
                distance = Point(point).distance(Point(las_point[:2]))
                result.append((idx, distance))
        return sorted(result, key=lambda p: p[1])

    z_min = float(las_data.header.z_min)
    z_range = float(las_data.header.z_max) - z_min
    if z_range < min_z_range:
        # 高低差10mは入るようにする。
        # 最高最低を範囲にすると、ほんの数cmで黒から白に変わっちゃったりする
        z_min = z_min - (min_z_range - z_range) / 2
        z_range = min_z_range

    def _calculate_weight(
        sorted_distances: list[float], max_use_count: int = 3
    ) -> list[float]:
        if len(sorted_distances) == 0:
            return []
        if sorted_distances[0] == 0:
            return [1.0]

        reciprocals = []
        for i in range(min(max_use_count, len(sorted_distances))):
            distance = sorted_distances[i]
            reciprocals.append(1 / (distance * distance))
        total = sum(reciprocals)
        return [reciprocal / total for reciprocal in reciprocals]

    def get_image(
        size: tuple[int, int], bounds: GeoBounds
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8], GeoBounds]:
        width, height = size
        # canvas_boundsの範囲をキャンバスいっぱいの画像にする
        # 端っこまで入れるために(width -1)*canvas_x_stepが確実にbounds.widthを超えるようにする
        canvas_x_step = (bounds.width + bounds.width / width) / (width - 1)
        canvas_y_step = (bounds.height + bounds.height / height) / (height - 1)

        bgr_canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        depth_canvas = np.full((height, width), 255, dtype=np.uint8)

        for canvas_j in range(height):
            for canvas_i in range(width):
                x = bounds.left + canvas_i * canvas_x_step
                y = bounds.top - canvas_j * canvas_y_step
                if outline is not None and not outline.polygon.contains(Point(x, y)):
                    continue
                las_i = int((x - las_x_min) / las_x_step) if las_x_step != 0 else 0
                las_j = int((las_y_max - y) / las_y_step) if las_y_step != 0 else 0
                las_indices = find_near_las_indices(
                    (x, y),
                    range(las_i - search_range, las_i + search_range),
                    range(las_j - search_range, las_j + search_range),
                )
                if len(las_indices) == 0:
                    continue
                # 距離の逆数の二乗に応じた加重平均
                weights = _calculate_weight(list(map(lambda p: p[1], las_indices)))
                blue = 0.0
                green = 0.0
                red = 0.0
                depth = 0.0
                for i in range(len(weights)):
                    idx, _ = las_indices[i]
                    blue += weights[i] * las_data.blue[idx]
                    green += weights[i] * las_data.green[idx]
                    red += weights[i] * las_data.red[idx]
                    depth += weights[i] * (las_data.z[idx] - z_min) / z_range
                bgr_canvas[canvas_j, canvas_i] = np.array(
                    [blue, green, red], dtype=np.uint8
                )
                depth_canvas[canvas_j, canvas_i] = int(depth * 255)
        return bgr_canvas, depth_canvas, bounds

    if canvas_bounds is not None:
        # canvas_boundsが与えられているときは、
        # canvas_sizeいっぱいにその範囲が描画されているはずなので合わせる
        return get_image(canvas_size, canvas_bounds)

    # canvas_boundsが渡されていないときは、まずcanvas_sizeを、
    # 最初のcanvas_sizeいっぱいにlas_boundsを入れられる大きさに変える
    if las_y_step < las_x_step:
        las_normal_height = las_height
        las_normal_width = las_height * las_bounds.width / las_bounds.height
    else:
        las_normal_height = las_width * las_bounds.height / las_bounds.width
        las_normal_width = las_width

    canvas_width, canvas_height = canvas_size
    if las_normal_width / las_normal_height < canvas_width / canvas_height:
        canvas_width = int(canvas_height * las_normal_width / las_normal_height)
    else:
        canvas_height = int(canvas_width * las_normal_height / las_normal_width)
    if canvas_width > max_factor * las_normal_width:
        canvas_width = int(max_factor * las_normal_width)
        canvas_height = int(max_factor * las_normal_height)

    return get_image((canvas_width, canvas_height), las_bounds)


def _points_to_grid(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    点群を格子状に構成します。
    入力点群から一意のX座標とY座標を列挙して格子を切ります。
    (X座標, Y座標)が一致する点が入力点群に存在する場合はその点が格子点に配置されます。
    存在しない場合は格子点はNoneになります。

    points: 3つの値（x座標とy座標とz座標）を持つ点の配列。
    return:
        - 行と列がそれぞれ入力点群のどの位置かを示す配列。
          この配列をあとすると、xがi番目、yがj番目の点はpoints[a[j,i]]
        - ソートされたx値の列。
        - ソートされたy値の列。
    """
    # X座標とY座標の一意な値を取得
    unique_x = np.unique(points[:, 0])
    unique_y = np.unique(points[:, 1])

    def fill_value(values):
        # 最小間隔を計算
        diffs = np.diff(values)
        min_diff = np.min(diffs[diffs > 0])

        # 補完
        new_values = [values[0]]
        for i in range(1, len(values)):
            diff = values[i] - values[i - 1]
            if diff > min_diff:
                steps = int(round(diff / min_diff))
                for j in range(1, steps):
                    new_values.append(values[i - 1] + j * min_diff)
            new_values.append(values[i])
        return np.array(new_values)

    # 格子の切れ目の位置
    x_sorted = fill_value(np.sort(unique_x))
    # 配列の先頭が画像の先頭になるように
    y_sorted = fill_value(np.sort(unique_y))[::-1]

    # 参照辞書を作成する
    x_to_index = {}
    for i, x in enumerate(x_sorted):
        x_to_index[x] = i
    y_to_index = {}
    for i, y in enumerate(y_sorted):
        y_to_index[y] = i

    # 格子を作成する
    grid_indices = np.empty((len(y_sorted), len(x_sorted)), dtype=int)
    grid_indices.fill(-1)

    for i, point in enumerate(points):
        x_index = x_to_index[point[0]]
        y_index = y_to_index[point[1]]
        grid_indices[y_index, x_index] = i

    return grid_indices, x_sorted, y_sorted
