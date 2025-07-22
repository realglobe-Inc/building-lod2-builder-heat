import math
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from numpy.typing import NDArray

from building_lod2_builder_heat.bounds import GeoBounds
from building_lod2_builder_heat.outline import GeoOutline


def load_ortho(
    ortho_file_path: Path,
    canvas_size: tuple[int, int],
    max_factor: float = 4.0,
    outline: GeoOutline | None = None,
) -> tuple[NDArray[np.uint8] | None, GeoBounds | None]:
    """
    オルソ画像を読み込む。

    指定されたファイルパスからオルソ画像を読み込み、
    目的の画像サイズ、オプションの幾何学的輪郭、
    縮尺係数などのパラメータに従って処理します。

    読み込んだオルソ画像がcanvas_sizeより大きいときは、
    キャンバスに収まるようにアスペクト比を維持しながら縮小します。
    canvas_sizeより小さいときは、キャンバスがらはみ出さないように、
    アスペクト比を維持してキャンバスの大きさまで拡大します。
    ただし、最大でもmax_factorまでの拡大とします。
    outlineがNoneでない場合は、画像を拡大するときに、
    outlineに沿ってジャギーの発生を抑制します。

    :param ortho_file_path: 読み込むオルソ画像のファイルパス。
    :param canvas_size: 出力画像を収める最大サイズ。
    :param max_factor: 拡大時の最大倍率。
    :param outline: 拡大する際に対象物の輪郭を保つための補助とする対象物の外形線。
    :return: 処理を経たオルソ画像と座標範囲。
    """
    with rasterio.open(ortho_file_path) as ortho:
        # 画像データを読み込む
        red = ortho.read(1)  # バンド1（赤）
        green = ortho.read(2)  # バンド2（緑）
        blue = ortho.read(3)  # バンド3（青）
        image = np.stack([blue, green, red], axis=-1)

        # 画像のサイズを取得する
        width = ortho.width
        height = ortho.height

        # CRSと範囲を取得します
        crs = ortho.crs
        if not crs:
            print(f"{ortho_file_path}の座標系が不明です", file=sys.stderr)
            return None, None
        bounds = ortho.bounds
        geo_bounds = GeoBounds(
            bounds.left, bounds.top, bounds.right, bounds.bottom, crs
        )

        # スケーリング係数を計算する
        canvas_width, canvas_height = canvas_size
        scale_w = canvas_width / width
        scale_h = canvas_height / height
        scale = min(scale_w, scale_h, max_factor)
        new_width = int(width * scale)
        new_height = int(height * scale)
        new_size = (new_width, new_height)

        if scale == 1:
            return image, geo_bounds
        if scale < 1 or outline is None:
            image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
            )
            return image, geo_bounds

        # 画像とマスクのスケーリング
        scaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)

        index_polygon = []
        for i in range(len(outline.polygon.exterior.coords)):
            x, y = outline.polygon.exterior.coords[i]
            x_idx = math.floor((x - geo_bounds.left) / geo_bounds.width * new_width)
            y_idx = math.floor((geo_bounds.top - y) / geo_bounds.height * new_height)
            index_polygon.append((x_idx, y_idx))

        # 凹形状を塗りつぶし
        mask = np.zeros((new_height, new_width), dtype=np.uint8)
        polygon_array = np.array(index_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [polygon_array], 255)
        scaled_image[mask == 0] = [255, 255, 255]
        image = scaled_image

        return image, geo_bounds
