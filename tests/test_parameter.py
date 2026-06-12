from __future__ import annotations

import json
from pathlib import Path

from building_lod2_builder_heat.common.parameter import (
    load_parameter,
    update_parameters,
)


def test_load_parameter_reads_whole_file_and_key(tmp_path: Path) -> None:
    """
    パラメータファイル全体と指定キーの値を読み取る。
    """
    param_file_path = tmp_path / "params.json"
    param_file_path.write_text('{"x": 1}', encoding="utf-8")

    assert load_parameter(param_file_path) == {"x": 1}
    assert load_parameter(param_file_path, "x") == 1
    assert load_parameter(param_file_path, "missing") is None


def test_load_parameter_key_from_non_object_returns_none(tmp_path: Path) -> None:
    """
    JSON オブジェクトでないファイルからキー指定で読み取らない。
    """
    param_file_path = tmp_path / "params.json"
    param_file_path.write_text("[1, 2]", encoding="utf-8")

    assert load_parameter(param_file_path, "x") is None


def test_load_parameter_invalid_json_returns_none(tmp_path: Path) -> None:
    """
    不正な JSON ファイルは読み取り失敗として扱う。
    """
    param_file_path = tmp_path / "params.json"
    param_file_path.write_text("{bad", encoding="utf-8")

    assert load_parameter(param_file_path, "x") is None


def test_update_parameters_overwrites_non_object_json(tmp_path: Path) -> None:
    """
    既存ファイルが JSON オブジェクトでない場合は全体を上書きする。
    """
    param_file_path = tmp_path / "params.json"
    param_file_path.write_text("[1, 2]", encoding="utf-8")

    update_parameters(param_file_path, {"x": 1})

    assert json.loads(param_file_path.read_text(encoding="utf-8")) == {"x": 1}
