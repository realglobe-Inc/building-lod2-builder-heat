import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterator

import pytest
from typer.testing import CliRunner

from building_lod2_builder_heat.commands.infer.main import app


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
    def test_run_dsm(
        self, runner, checkpoint_file, test_data_dir, answers_dir, temp_output_dir
    ):
        """DSMだけの場合のテスト"""
        dsm_dir = test_data_dir / "dsm"
        expected_answers_dir = answers_dir / "dsm"

        output_dir = temp_output_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            app,
            [
                str(checkpoint_file),
                str(dsm_dir),
                str(output_dir),
                "--overwrite",
            ],
        )

        assert result.exit_code == 0, f"コマンドが失敗しました: {result.output}"

        for las_file in dsm_dir.glob("*.las"):
            output_file = output_dir / f"{las_file.stem}.json"
            expected_file = expected_answers_dir / f"{las_file.stem}.json"

            assert json.loads(output_file.read_text()) == json.loads(
                expected_file.read_text()
            )

    @pytest.mark.integration
    def test_run_dsm_ortho(
        self,
        runner: CliRunner,
        checkpoint_file: Path,
        test_data_dir: Path,
        answers_dir: Path,
        temp_output_dir: Path,
    ):
        """オルソ画像を追加した場合のテスト"""
        dsm_dir = test_data_dir / "dsm"
        ortho_dir = test_data_dir / "ortho_clipped"
        expected_answers_dir = answers_dir / "dsm_ortho_clipped"

        output_dir = temp_output_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            app,
            [
                str(checkpoint_file),
                str(dsm_dir),
                str(output_dir),
                "--ortho-dir",
                str(ortho_dir),
                "--overwrite",
            ],
        )

        assert result.exit_code == 0, f"コマンドが失敗しました: {result.output}"

        for las_file in dsm_dir.glob("*.las"):
            output_file = output_dir / f"{las_file.stem}.json"
            expected_file = expected_answers_dir / f"{las_file.stem}.json"

            assert json.loads(output_file.read_text()) == json.loads(
                expected_file.read_text()
            )

    @pytest.mark.integration
    def test_run_dsm_obj(
        self,
        runner: CliRunner,
        checkpoint_file: Path,
        test_data_dir: Path,
        answers_dir: Path,
        temp_output_dir: Path,
    ):
        """外形線を追加した場合のテスト"""
        dsm_dir = test_data_dir / "dsm"
        obj_dir = test_data_dir / "obj"
        expected_answers_dir = answers_dir / "dsm_obj"

        output_dir = temp_output_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            app,
            [
                str(checkpoint_file),
                str(dsm_dir),
                str(output_dir),
                "--obj-dir",
                str(obj_dir),
                "--overwrite",
            ],
        )

        assert result.exit_code == 0, f"コマンドが失敗しました: {result.output}"

        for las_file in dsm_dir.glob("*.las"):
            output_file = output_dir / f"{las_file.stem}.json"
            expected_file = expected_answers_dir / f"{las_file.stem}.json"

            assert json.loads(output_file.read_text()) == json.loads(
                expected_file.read_text()
            )

    @pytest.mark.integration
    def test_run_dsm_ortho_obj(
        self,
        runner: CliRunner,
        checkpoint_file: Path,
        test_data_dir: Path,
        answers_dir: Path,
        temp_output_dir: Path,
    ):
        """オルソ画像と外形線を追加した場合のテスト"""
        dsm_dir = test_data_dir / "dsm"
        ortho_dir = test_data_dir / "ortho"
        obj_dir = test_data_dir / "obj"
        expected_answers_dir = answers_dir / "dsm_ortho_obj"

        output_dir = temp_output_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            app,
            [
                str(checkpoint_file),
                str(dsm_dir),
                str(output_dir),
                "--ortho-dir",
                str(ortho_dir),
                "--obj-dir",
                str(obj_dir),
                "--overwrite",
            ],
        )

        assert result.exit_code == 0, f"コマンドが失敗しました: {result.output}"

        for las_file in dsm_dir.glob("*.las"):
            output_file = output_dir / f"{las_file.stem}.json"
            expected_file = expected_answers_dir / f"{las_file.stem}.json"

            assert json.loads(output_file.read_text()) == json.loads(
                expected_file.read_text()
            )

    @pytest.mark.integration
    def test_run_downsample_dsm(
        self,
        runner: CliRunner,
        checkpoint_file: Path,
        test_data_dir: Path,
        answers_dir: Path,
        temp_output_dir: Path,
    ):
        """間引いたDSMの場合のテスト"""
        dsm_dir = test_data_dir / "dsm_downsample"
        expected_answers_dir = answers_dir / "dsm_downsample"

        output_dir = temp_output_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            app,
            [
                str(checkpoint_file),
                str(dsm_dir),
                str(output_dir),
                "--overwrite",
            ],
        )

        assert result.exit_code == 0, f"コマンドが失敗しました: {result.output}"

        for las_file in dsm_dir.glob("*.las"):
            output_file = output_dir / f"{las_file.stem}.json"
            expected_file = expected_answers_dir / f"{las_file.stem}.json"

            assert json.loads(output_file.read_text()) == json.loads(
                expected_file.read_text()
            )
