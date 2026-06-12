from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset

from building_lod2_builder_heat.common import file_names, parameter_keys
from building_lod2_builder_heat.common.parameter import load_parameter


class RooflineSampleInfo(TypedDict):
    """
    屋根線抽出の入力ファイル情報を表す型。
    """

    building_id: str
    rgb_file_path: Path
    depth_file_path: Path
    output_dir_path: Path


class RooflineSample(TypedDict):
    """
    屋根線抽出の1サンプルを表す型。
    """

    building_id: str
    input_rgb: NDArray[np.uint8]
    input_depth: NDArray[np.generic]
    bgr_image: NDArray[np.uint8]
    rgb_file_path: Path
    depth_file_path: Path
    output_dir_path: Path


class RooflineDataset(Dataset[RooflineSample]):
    """
    屋根線抽出のためのデータセット。
    """

    def __init__(
        self,
        data_root_dir_path: Path,
        output_root_dir_path: Path,
        skip_exist: bool = True,
    ) -> None:
        """
        :param data_root_dir_path: データディレクトリのパス。
        :param output_root_dir_path: 出力ディレクトリのパス。
        :param skip_exist: 既に結果が存在する場合はスキップするかどうか。
        """
        self.data_root_dir_path = data_root_dir_path
        self.output_root_dir_path = output_root_dir_path
        self.skip_exist = skip_exist

        self.samples: list[RooflineSampleInfo] = []
        for input_dir_path in sorted(data_root_dir_path.iterdir()):
            if not input_dir_path.is_dir():
                continue

            building_id = input_dir_path.stem
            rgb_file_path = input_dir_path / file_names.ROOFLINE_EXTRACTION_INPUT_RGB
            depth_file_path = (
                input_dir_path / file_names.ROOFLINE_EXTRACTION_INPUT_DEPTH
            )

            if not rgb_file_path.exists() or not depth_file_path.exists():
                continue

            output_dir_path = output_root_dir_path / building_id
            output_params_file_path = (
                output_dir_path / file_names.EXTRACT_ROOFLINE_OUTPUT
            )

            if self.skip_exist and (
                load_parameter(output_params_file_path, parameter_keys.ROOFLINE_EDGES)
                is not None
            ):
                continue

            self.samples.append(
                {
                    "building_id": building_id,
                    "rgb_file_path": rgb_file_path,
                    "depth_file_path": depth_file_path,
                    "output_dir_path": output_dir_path,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RooflineSample:
        sample_info = self.samples[idx]
        rgb_file_path = sample_info["rgb_file_path"]
        depth_file_path = sample_info["depth_file_path"]

        input_rgb, input_depth = load_roofline_images(rgb_file_path, depth_file_path)

        # HEATモデルはBGR形式を期待する
        bgr_image = np.ascontiguousarray(input_rgb[:, :, [2, 1, 0]])

        return {
            "building_id": sample_info["building_id"],
            "input_rgb": input_rgb,
            "input_depth": input_depth,
            "bgr_image": bgr_image,
            "rgb_file_path": rgb_file_path,
            "depth_file_path": depth_file_path,
            "output_dir_path": sample_info["output_dir_path"],
        }


def load_roofline_images(
    rgb_file_path: Path, depth_file_path: Path
) -> tuple[NDArray[np.uint8], NDArray[np.generic]]:
    """
    屋根線抽出用の RGB 画像と depth 画像を読み込む。

    :param rgb_file_path: RGB画像のパス。
    :param depth_file_path: Depth画像のパス。
    :returns: RGB画像と depth 画像。
    :raises ValueError: depth画像が単一チャンネルでない、または画像サイズが
        一致しない場合。
    """
    with Image.open(rgb_file_path) as img:
        input_rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    with Image.open(depth_file_path) as img:
        input_depth = np.array(img)

    if input_depth.ndim != 2:
        raise ValueError(f"{depth_file_path} は単一チャンネル画像ではありません")

    if input_rgb.shape[:2] != input_depth.shape[:2]:
        raise ValueError(
            f"{rgb_file_path} と {depth_file_path} の画像サイズが一致しません: "
            f"{input_rgb.shape[1]}x{input_rgb.shape[0]} != "
            f"{input_depth.shape[1]}x{input_depth.shape[0]}"
        )

    return input_rgb, input_depth
