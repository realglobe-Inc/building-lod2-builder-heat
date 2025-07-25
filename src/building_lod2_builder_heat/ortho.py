import sys
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio import MemoryFile
from rasterio.enums import Resampling
from rasterio.mask import mask

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
        data = ortho.read()
        image = np.transpose(data, (1, 2, 0))[:, :, [2, 1, 0]]

        # 画像のサイズを取得する
        width = ortho.width
        height = ortho.height

        # CRSと範囲を取得します
        crs = ortho.crs
        if not crs:
            print(f"{ortho_file_path}の座標系が不明です", file=sys.stderr)
            return None, None
        outline = outline.transform_to(crs) if outline else None
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

        if scale == 1:
            return image, geo_bounds

        data = ortho.read(
            out_shape=(ortho.count, new_height, new_width),
            resampling=Resampling.lanczos,
        )

        if scale < 1 or outline is None:
            return np.transpose(data, (1, 2, 0)), geo_bounds

        scale_transform = rasterio.transform.from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top, new_width, new_height
        )
        meta = ortho.meta.copy()
        meta.update(
            {
                "transform": scale_transform,
                "crs": crs,
                "width": new_width,
                "height": new_height,
            }
        )

        with MemoryFile() as tmp_file:
            with tmp_file.open(**meta) as tmp_ortho:
                tmp_ortho.write(data)

                clipped_data, clipped_transform = mask(
                    tmp_ortho, [outline.polygon], filled=True, nodata=255
                )
                clipped_meta = tmp_ortho.meta.copy()
                clipped_meta.update(
                    {
                        "transform": clipped_transform,
                        "crs": crs,
                        "width": clipped_data.shape[2],
                        "height": clipped_data.shape[1],
                    }
                )

                bounds = tmp_ortho.bounds
                geo_bounds = GeoBounds(
                    bounds.left, bounds.top, bounds.right, bounds.bottom, crs
                )

                return np.transpose(clipped_data, (1, 2, 0)), geo_bounds
