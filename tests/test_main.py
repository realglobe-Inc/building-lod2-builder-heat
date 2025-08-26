import json
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterator

import pytest
from typer.testing import CliRunner

from building_lod2_builder_heat.commands.extract_roofline import file_names
from building_lod2_builder_heat.commands.extract_roofline.main import app


class TestRunIntegration:
    """runコマンドの統合テスト"""

    @pytest.fixture
    def project_root(self) -> Path:
        """プロジェクトルートを取得"""
        return Path(__file__).parent.parent

    @pytest.fixture
    def test_data_dir(self, project_root: Path) -> Path:
        """テストデータディレクトリを取得"""
        return project_root / "test_data"

    @pytest.fixture
    def answers_dir(self, test_data_dir: Path) -> Path:
        """期待値ディレクトリを取得"""
        return test_data_dir / "answers"

    @pytest.fixture
    def temp_output_dir(self) -> Iterator[Path]:
        """一時出力ディレクトリを作成"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def checkpoint_file(self, project_root: Path, test_data_dir: Path) -> Path:
        """チェックポイントファイルを用意"""
        checkpoint_file = project_root / "roof_edge_detection_parameter.pth"
        if checkpoint_file.exists():
            return checkpoint_file

        checkpoint_file = test_data_dir / "roof_edge_detection_parameter.pth"
        if checkpoint_file.exists():
            return checkpoint_file

        url = "https://github.com/realglobe-Inc/bldg-lod2-tool/releases/download/PretrainedModels-1.0/roof_edge_detection_parameter.pth"
        urllib.request.urlretrieve(url, checkpoint_file)
        return checkpoint_file

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Typer CLI runner"""
        return CliRunner()

    @pytest.mark.integration
    def test_all(
        self, runner, checkpoint_file, test_data_dir, answers_dir, temp_output_dir
    ):
        input_root_dir = test_data_dir / "input"
        output_root_dir = temp_output_dir

        result = runner.invoke(
            app,
            [
                str(checkpoint_file),
                str(input_root_dir),
                "--output-dir",
                str(output_root_dir),
            ],
        )

        input_dirs = list(input_root_dir.iterdir())
        for input_dir in sorted(input_dirs):
            err_msg = (input_dir.name, result.output)

            output_dir = output_root_dir / input_dir.name
            assert output_dir.exists(), err_msg
            output_files = list(output_dir.iterdir())
            output_file = output_dir / file_names.PARAMETERS
            assert set(output_files) == {output_file}, err_msg

            expected_answers_dir = answers_dir / input_dir.name
            expected_file = expected_answers_dir / output_file.name
            assert_json_files_eq(
                output_file, expected_file, err_msg, exclude=r".*_source_.*"
            )


def assert_json_files_eq(
    actual_file: Path, expected_file: Path, err_msg: object, exclude: str = None
):
    with open(actual_file, "r") as f:
        actual_json = json.load(f)
    with open(expected_file, "r") as f:
        expected_json = json.load(f)
    if exclude:
        actual_json = {k: v for k, v in actual_json.items() if not re.match(exclude, k)}
        expected_json = {
            k: v for k, v in expected_json.items() if not re.match(exclude, k)
        }
    assert actual_json == expected_json, err_msg
