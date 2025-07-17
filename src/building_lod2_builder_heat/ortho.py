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
) -> tuple[NDArray[np.uint8], GeoBounds] | None:
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
        image = ortho.read(1)

        # 画像のサイズを取得する
        width = ortho.width
        height = ortho.height

        # CRSと範囲を取得します
        crs = ortho.crs
        if not crs:
            print(f"{ortho_file_path}の座標系が不明です", file=sys.stderr)
            return None
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

        # 外形線からマスクを作成する
        mask = np.zeros((height, width), dtype=np.uint8)
        outline_coords = np.array(outline.polygon.exterior.coords).astype(np.int32)
        # TODO 座標からインデクスへの変換

        cv2.fillPoly(mask, [outline_coords], 255)

        # 画像とマスクのスケーリング
        scaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
        scaled_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_LANCZOS4)

        # 外形線にアンチエイリアシングを適用する
        edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        edges = cv2.filter2D(scaled_mask, -1, edge_kernel)
        edges = edges > 0

        # 滑らかな縁に
        scaled_image[edges] = cv2.GaussianBlur(scaled_image, (3, 3), 0)[edges]
        image = scaled_image

        return image, geo_bounds
