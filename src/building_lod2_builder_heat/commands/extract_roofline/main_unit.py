from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from heat import HEAT
from loguru import logger
from numpy.typing import NDArray

from building_lod2_builder_heat.common import file_names, parameter_keys
from building_lod2_builder_heat.common.parameter import (
    update_parameters,
)


def main_unit(
    rgb_file_path: Path,
    depth_file_path: Path,
    model: HEAT,
    output_dir_path: Path,
    byproduct_dir_path: Path | None = None,
    backup: bool = False,
):
    input_rgb = np.array(Image.open(rgb_file_path))
    input_depth = np.array(Image.open(depth_file_path))

    # TODO depthの利用
    corners, edges = model.infer(input_rgb[:, :, [2, 1, 0]])  # RGB -> BGR

    logger.info(f"検出結果: {len(corners)}個の角 {len(edges)}個の辺")

    output_param_file_path = output_dir_path / file_names.EXTRACT_ROOFLINE_OUTPUT
    if backup:
        _backup_file(output_param_file_path)
    params = {
        parameter_keys.ROOFLINE_CORNERS: corners.tolist(),
        parameter_keys.ROOFLINE_EDGES: edges.tolist(),
        parameter_keys.SOURCE_RGB: str(rgb_file_path),
        parameter_keys.SOURCE_DEPTH: str(depth_file_path),
    }
    update_parameters(output_param_file_path, params)

    # 結果画像を出力する
    if byproduct_dir_path is not None:
        rgb_out = byproduct_dir_path / file_names.ROOFLINE_EXTRACTION_RESULT_RGB
        depth_out = byproduct_dir_path / file_names.ROOFLINE_EXTRACTION_RESULT_DEPTH
        if backup:
            _backup_file(rgb_out)
            _backup_file(depth_out)
        visualized_rgb = _visualize_detection_results(input_rgb, corners, edges)
        Image.fromarray(visualized_rgb).save(rgb_out)
        visualized_depth = _visualize_detection_results(
            np.array(Image.fromarray(input_depth).convert("RGB")), corners, edges
        )
        Image.fromarray(visualized_depth).save(depth_out)


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


def _backup_file(file_path: Path):
    """
    file_pathとしてファイルを保存する前に、
    同じファイル名があったら古いファイルをバックアップする
    """
    if not file_path.exists():
        return
    date_tag = datetime.now().strftime("%Y-%m-%d")
    i = 0
    while True:
        backup_file_path = file_path.with_stem(f"{file_path.stem}_{date_tag}.{i}")
        if not backup_file_path.exists():
            file_path.rename(backup_file_path)
            return
        i += 1
