from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from heat import HEAT
from loguru import logger
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from building_lod2_builder_heat.commands.extract_roofline.dataset import (
    load_roofline_images,
)
from building_lod2_builder_heat.common import file_names, parameter_keys
from building_lod2_builder_heat.common.parameter import update_parameters


def main_unit(
    rgb_file_path: Path,
    depth_file_path: Path,
    model: HEAT,
    output_dir_path: Path,
    byproduct_dir_path: Path | None = None,
    backup: bool = False,
) -> None:
    """
    1枚の画像に対して屋根線を抽出し、結果を保存する。

    :param rgb_file_path: RGB画像のパス。
    :param depth_file_path: Depth画像のパス。
    :param model: 推論に使用するモデル。
    :param output_dir_path: 出力先ディレクトリ。
    :param byproduct_dir_path: 副産物（可視化画像）の出力先ディレクトリ。
    :param backup: 既に結果が存在する場合にバックアップを作成するかどうか。
    """
    input_rgb, input_depth = load_roofline_images(rgb_file_path, depth_file_path)

    # TODO depthの利用
    corners, edges = model.infer(
        np.ascontiguousarray(input_rgb[:, :, [2, 1, 0]])
    )  # RGB -> BGR

    save_roofline_result(
        corners=corners,
        edges=edges,
        rgb_file_path=rgb_file_path,
        depth_file_path=depth_file_path,
        input_rgb=input_rgb,
        input_depth=input_depth,
        output_dir_path=output_dir_path,
        byproduct_dir_path=byproduct_dir_path,
        backup=backup,
    )


def save_roofline_result(
    corners: NDArray[np.float64],
    edges: NDArray[np.int32],
    rgb_file_path: Path,
    depth_file_path: Path,
    input_rgb: NDArray[np.uint8],
    input_depth: NDArray[np.generic],
    output_dir_path: Path,
    byproduct_dir_path: Path | None = None,
    backup: bool = False,
) -> None:
    """
    推論結果をファイルに保存し、必要に応じて可視化画像を出力する。

    :param corners: 推論されたコーナー座標。
    :param edges: 推論されたエッジ。
    :param rgb_file_path: 入力RGB画像のパス（メタデータ用）。
    :param depth_file_path: 入力Depth画像のパス（メタデータ用）。
    :param input_rgb: 入力RGB画像データ。
    :param input_depth: 入力Depth画像データ。
    :param output_dir_path: 出力先ディレクトリ。
    :param byproduct_dir_path: 副産物（可視化画像）の出力先ディレクトリ。
    :param backup: 既に結果が存在する場合にバックアップを作成するかどうか。
    """
    logger.debug(f"{len(corners)}個の角、{len(edges)}個の辺を検出しました")

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
        rgb_out = byproduct_dir_path / file_names.EXTRACT_ROOFLINE_RESULT_RGB
        depth_out = byproduct_dir_path / file_names.EXTRACT_ROOFLINE_RESULT_DEPTH
        if backup:
            _backup_file(rgb_out)
            _backup_file(depth_out)
        visualized_rgb = _visualize_detection_results(input_rgb, corners, edges)
        Image.fromarray(visualized_rgb).save(rgb_out)
        visualized_depth = _visualize_detection_results(
            np.array(Image.fromarray(input_depth).convert("RGB")), corners, edges
        )
        Image.fromarray(visualized_depth).save(depth_out)

    logger.info(f"{output_dir_path} に出力しました")


def _visualize_detection_results(
    image: NDArray[np.uint8], corners: NDArray[np.float64], edges: NDArray[np.int32]
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


def _backup_file(file_path: Path) -> None:
    """
    file_path としてファイルを保存する前に既存ファイルをバックアップする。

    :param file_path: バックアップ対象のファイルパス。
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
