from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path

from loguru import logger


def load_parameter(param_file_path: Path, key: str | None = None) -> object | None:
    """
    パラメータファイルから値を読み取る。

    :param param_file_path: パラメータファイルのパス。
    :param key: 読み取るキー。``None`` の場合は JSON 全体を返す。
    :returns: 読み取った値。ファイルが無い、JSON が不正、またはキーが無い場合は
        ``None``。
    """
    try:
        with param_file_path.open(encoding="utf-8") as f:
            json_data = json.load(f)
            if key is None:
                return json_data
            if not isinstance(json_data, dict):
                return None
            return json_data.get(key)
    except FileNotFoundError, JSONDecodeError:
        return None


def update_parameters(
    param_file_path: Path, params: dict[str, object], overwrite: bool = False
) -> None:
    """
    パラメータファイルに書き込む。

    :param param_file_path: パラメータファイルのパス。
    :param params: 更新するデータ。
    :param overwrite: 既存の内容を引き継がないか。
    """
    merged_params = params.copy()

    # 既存のデータを読み込み
    if not overwrite and param_file_path.is_file():
        try:
            with param_file_path.open(encoding="utf-8") as f:
                old = json.load(f)
        except JSONDecodeError as e:
            logger.warning(
                f"{param_file_path}をJSONとして読み込めなかったため、"
                f"全体を上書きします: {e}",
            )
            param_file_path.unlink()
        else:
            if isinstance(old, dict):
                old.update(params)
                merged_params = old
            else:
                logger.warning(
                    f"{param_file_path}がJSONオブジェクトでないため、全体を上書きします"
                )

    # ファイルに書き戻し
    param_file_path.parent.mkdir(parents=True, exist_ok=True)
    with param_file_path.open("w", encoding="utf-8") as f:
        json.dump(merged_params, f, ensure_ascii=False, indent=2)
