import json
from json import JSONDecodeError
from pathlib import Path

from loguru import logger


def load_parameter(param_file_path: Path, key: str) -> object:
    try:
        with open(param_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            return json_data.get(key)
    except (FileNotFoundError, JSONDecodeError):
        return None


def update_parameters(param_file_path: Path, params: dict, overwrite: bool = False):
    """
    パラメータファイルに書き込む。

    :param param_file_path: パラメータファイルのパス
    :param params: 更新するデータ
    :param overwrite: 既存の内容を引き継がないか
    """
    # 既存のデータを読み込み
    if not overwrite and param_file_path.exists():
        try:
            with open(param_file_path, "r", encoding="utf-8") as f:
                old = json.load(f)
                old.update(params)
                params = old
        except JSONDecodeError as e:
            logger.info(
                f"{param_file_path}をJSONとして読み込めなかったため、全体を上書きします: {e}",
            )
            param_file_path.unlink()

    # ファイルに書き戻し
    param_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(param_file_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
