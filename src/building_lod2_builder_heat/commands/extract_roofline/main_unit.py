from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from heat import HEAT
from numpy.typing import NDArray

from building_lod2_builder_heat.commands.extract_roofline import (
    file_names,
    parameter_keys,
)
from building_lod2_builder_heat.commands.extract_roofline.bounds import Bounds
from building_lod2_builder_heat.commands.extract_roofline.dsm import load_dsm
from building_lod2_builder_heat.commands.extract_roofline.obj import Obj3D
from building_lod2_builder_heat.commands.extract_roofline.ortho import load_ortho
from building_lod2_builder_heat.commands.extract_roofline.outline import GeoOutline
from building_lod2_builder_heat.commands.extract_roofline.parameter import (
    load_crs_parameter,
    update_parameters,
)


def main_unit(
    model: HEAT,
    dsm_file_path: Path,
    canvas_size: int,
    output_dir_path: Path,
    param_file_path: Path | None = None,
    obj_file_path: Path | None = None,
    ortho_file_path: Path | None = None,
    byproduct_dir_path: Path | None = None,
):
    crs = load_crs_parameter(param_file_path)

    outline: GeoOutline | None = None
    if obj_file_path is not None:
        if crs is None:
            raise ValueError(f"{param_file_path}から座標系を読み込めませんでした")

        outline_polygon = Obj3D.load(obj_file_path).calculate_horizontal_outline()
        print(
            f"{obj_file_path}から{len(outline_polygon.exterior.coords)}頂点の外形線を読み込みました"
        )
        outline = GeoOutline(outline_polygon, crs)

    ortho_rgb: NDArray[np.uint8] | None = None
    ortho_bounds: Bounds | None = None
    if ortho_file_path is not None:
        ortho_rgb, ortho_bounds = load_ortho(
            ortho_file_path,
            canvas_size=(canvas_size, canvas_size),
            outline=outline,
            default_crs=crs,
        )

        print(f"{ortho_file_path}からRGB画像を読み込みました")

        # 入力画像を出力する
        if byproduct_dir_path is not None:
            Image.fromarray(ortho_rgb).save(
                byproduct_dir_path / file_names.ROOFLINE_EXTRACTION_INPUT_ORTHO_RGB
            )

    dsm_rgb, dsm_depth, dsm_bounds = load_dsm(
        dsm_file_path,
        canvas_size=(
            (canvas_size, canvas_size)
            if ortho_rgb is None
            else (ortho_rgb.shape[1], ortho_rgb.shape[0])
        ),
        outline=outline,
        canvas_bounds=ortho_bounds,
        default_crs=crs,
    )
    print(f"{dsm_file_path}からRGB画像と深度を読み込みました")

    # 入力画像を出力する
    if byproduct_dir_path is not None:
        Image.fromarray(dsm_rgb).save(
            byproduct_dir_path / file_names.ROOFLINE_EXTRACTION_INPUT_DSM_RGB
        )
        Image.fromarray(dsm_depth, mode="L").save(
            byproduct_dir_path / file_names.ROOFLINE_EXTRACTION_INPUT_DSM_DEPTH
        )

    input_rgb = ortho_rgb if ortho_rgb is not None else dsm_rgb
    padded_rgb, bounds = _pad_image(input_rgb, canvas_size)
    padded_depth, _ = _pad_image(dsm_depth, canvas_size, pixel_size=1)

    # TODO depthの利用
    corners, edges = model.infer(padded_rgb[:, :, [2, 1, 0]])  # RGB -> BGR

    print(f"検出結果: {len(corners)}個の角 {len(edges)}個の辺")

    output_file_path = output_dir_path / file_names.PARAMETERS
    result_params = {
        parameter_keys.ROOFLINE_CORNERS: corners.tolist(),
        parameter_keys.ROOFLINE_EDGES: edges.tolist(),
        parameter_keys.ROOFLINE_EXTRACTION_INPUT_CANVAS_SIZE: canvas_size,
        parameter_keys.ROOFLINE_EXTRACTION_INPUT_IMAGE_BOUNDS: bounds.ltrb,
        parameter_keys.ROOFLINE_EXTRACTION_INPUT_GEO_BOUNDS: dsm_bounds.ltrb,
        parameter_keys.CRS: dsm_bounds.crs.to_string(),
    }
    update_parameters(output_file_path, result_params)

    # 結果画像を出力する
    if byproduct_dir_path is not None:
        visualized_rgb = _visualize_detection_results(padded_rgb, corners, edges)
        Image.fromarray(visualized_rgb).save(
            byproduct_dir_path / file_names.ROOFLINE_EXTRACTION_RESULT_RGB
        )
        visualized_depth = _visualize_detection_results(
            np.array(Image.fromarray(padded_depth).convert("RGB")), corners, edges
        )
        Image.fromarray(visualized_depth).save(
            byproduct_dir_path / file_names.ROOFLINE_EXTRACTION_RESULT_DEPTH
        )


def _pad_image(
    input_image: NDArray[np.uint8],
    canvas_size: int,
    bg_color: int = 255,
    pixel_size: int = 3,
) -> tuple[NDArray[np.uint8], Bounds]:
    """
    画像を正方形のキャンバスに収める。

    :param input_image: 入力画像
    :param canvas_size: キャンバスの大きさ
    :param bg_color: 広げた部分のグレーカラー
    :return: キャンバスの中央に配置された画像とその配置場所
    """

    image_height, image_width = input_image.shape[:2]

    # 中央寄せでコピー
    left = (canvas_size - image_width) // 2
    bottom = (canvas_size - image_height) // 2
    right = left + image_width
    top = bottom + image_height

    shape = (
        (canvas_size, canvas_size, pixel_size)
        if pixel_size > 1
        else (canvas_size, canvas_size)
    )
    canvas = np.full(shape, bg_color, dtype=np.uint8)
    canvas[bottom:top, left:right] = input_image
    bounds = Bounds(left, top, right, bottom)

    return canvas, bounds


def _visualize_detection_results(
    image: NDArray[np.uint8], corners: NDArray[np.int32], edges: NDArray[np.int32]
) -> NDArray[np.uint8]:
    """
    検出結果の可視化画像を生成する。

    :param image: 元のRGB画像
    :param corners: 角点座標の配列
    :param edges: エッジの配列

    :return: 結果を重ねたRGB画像
    """
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # エッジを描画（緑色の線）
    edge: NDArray[np.int32]
    for edge in edges:
        start_point = list(corners[edge[0]])
        end_point = list(corners[edge[1]])
        draw.line([start_point, end_point], fill=(0, 255, 0), width=1)

    d = 2
    # 角点を描画（赤色の円）
    for corner in corners:
        x, y = corner
        # 円を描画（半径dの円）
        draw.ellipse([x - d, y - d, x + d, y + d], fill=(0, 0, 255))

    return np.array(pil_image)
