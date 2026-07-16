from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from building_lod2_builder_heat.commands.extract_roofline import main as main_module
from building_lod2_builder_heat.commands.extract_roofline.dataset import (
    RooflineDataset,
)
from building_lod2_builder_heat.common import file_names, parameter_keys


class FailingHeat:
    """
    バッチ推論で失敗する HEAT テストダブル。
    """

    device = "cpu"

    def __init__(self, force_cpu: bool) -> None:
        """
        初期化引数を受け取る。

        :param force_cpu: CPU 強制指定。
        """
        self.force_cpu = force_cpu

    def load_checkpoint(self, checkpoint_file_path: Path) -> None:
        """
        チェックポイント読み込みを成功扱いにする。

        :param checkpoint_file_path: チェックポイントのパス。
        """

    def infer_batch(self, bgr_images: list[np.ndarray]) -> list[object]:
        """
        バッチ推論を失敗させる。

        :param bgr_images: BGR 画像のリスト。
        :raises RuntimeError: 常に発生するテスト用エラー。
        """
        raise RuntimeError("boom")


class SuccessfulHeat:
    """
    固定した角点と辺を返す HEAT テストダブル。
    """

    device = "cpu"

    def __init__(self, force_cpu: bool) -> None:
        """
        :param force_cpu: CPU 強制指定。
        """
        self.force_cpu = force_cpu

    def load_checkpoint(self, checkpoint_file_path: Path) -> None:
        """
        チェックポイント読み込みを成功扱いにする。

        :param checkpoint_file_path: チェックポイントのパス。
        """

    def infer_batch(
        self, bgr_images: list[np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        各画像に固定した推論結果を返す。

        :param bgr_images: BGR 画像のリスト。
        :returns: HEAT の画素index角点と辺。
        """
        return [
            (
                np.array([[0, 0], [3, 2]], dtype=np.int64),
                np.array([[0, 1]], dtype=np.int32),
            )
            for _ in bgr_images
        ]


def test_roofline_dataset_converts_rgb_to_three_channels(tmp_path: Path) -> None:
    """
    RGB 入力は 3 チャンネルへ正規化し、BGR 画像を作る。
    """
    input_dir_path = tmp_path / "building"
    _write_input_image(
        input_dir_path,
        rgb=np.full((2, 3, 4), [10, 20, 30, 40], dtype=np.uint8),
    )

    sample = RooflineDataset(tmp_path, tmp_path / "out", skip_exist=False)[0]

    assert sample["input_rgb"].shape == (2, 3, 3)
    assert sample["input_rgb"][0, 0].tolist() == [10, 20, 30]
    assert sample["bgr_image"][0, 0].tolist() == [30, 20, 10]


def test_roofline_dataset_ignores_directory_without_rgb(tmp_path: Path) -> None:
    """
    RGB 入力が存在しないディレクトリは処理対象にしない。
    """
    (tmp_path / "building").mkdir()

    dataset = RooflineDataset(tmp_path, tmp_path / "out", skip_exist=False)

    assert len(dataset) == 0


def test_cli_records_error_for_each_sample_when_batch_inference_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    バッチ推論の失敗を対象ごとの出力 JSON に記録する。
    """
    monkeypatch.setattr(main_module, "HEAT", FailingHeat)

    checkpoint_file_path = tmp_path / "checkpoint.pth"
    checkpoint_file_path.touch()
    input_root_dir_path = tmp_path / "input"
    output_root_dir_path = tmp_path / "output"
    for building_id in ("a", "b"):
        _write_input_image(input_root_dir_path / building_id)

    result = CliRunner().invoke(
        main_module.app,
        [
            str(checkpoint_file_path),
            str(input_root_dir_path),
            "--output-dir",
            str(output_root_dir_path),
            "--force-cpu",
            "--batch-size",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    for building_id in ("a", "b"):
        output_file_path = (
            output_root_dir_path / building_id / file_names.EXTRACT_ROOFLINE_OUTPUT
        )
        params = json.loads(output_file_path.read_text(encoding="utf-8"))
        assert params[parameter_keys.ERROR] == "boom"
        assert "RuntimeError: boom" in params[parameter_keys.TRACEBACK]
        assert parameter_keys.ROOFLINE_EDGES not in params


def test_cli_converts_heat_pixel_indices_to_continuous_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    HEAT の画素index角点は保存前に画素中心の連続座標へ変換する。
    """
    monkeypatch.setattr(main_module, "HEAT", SuccessfulHeat)

    checkpoint_file_path = tmp_path / "checkpoint.pth"
    checkpoint_file_path.touch()
    input_root_dir_path = tmp_path / "input"
    output_root_dir_path = tmp_path / "output"
    _write_input_image(input_root_dir_path / "building")

    result = CliRunner().invoke(
        main_module.app,
        [
            str(checkpoint_file_path),
            str(input_root_dir_path),
            "--output-dir",
            str(output_root_dir_path),
            "--force-cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    output_file_path = (
        output_root_dir_path / "building" / file_names.EXTRACT_ROOFLINE_OUTPUT
    )
    params = json.loads(output_file_path.read_text(encoding="utf-8"))
    assert params[parameter_keys.ROOFLINE_CORNERS] == [[0.5, 0.5], [3.5, 2.5]]
    assert params[parameter_keys.ROOFLINE_EDGES] == [[0, 1]]


def _write_input_image(
    input_dir_path: Path,
    rgb: np.ndarray | None = None,
) -> Path:
    """
    屋根線抽出用の入力画像を書き込む。

    :param input_dir_path: 入力ディレクトリ。
    :param rgb: RGB 画像配列。
    :returns: 書き込んだ RGB 画像のパス。
    """
    input_dir_path.mkdir(parents=True, exist_ok=True)
    rgb_file_path = input_dir_path / file_names.EXTRACT_ROOFLINE_PREPROCESS_RGB
    rgb_array = rgb if rgb is not None else np.zeros((4, 4, 3), dtype=np.uint8)
    Image.fromarray(rgb_array).save(rgb_file_path)
    return rgb_file_path
