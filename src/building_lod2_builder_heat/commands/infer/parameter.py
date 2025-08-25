import json
import sys
from json import JSONDecodeError
from pathlib import Path

from pyproj import CRS


def load_obj_crs_from_json(json_file_path: Path) -> CRS | None:
    return _load_crs_from_json(json_file_path, "obj_")


def load_dsm_crs_from_json(json_file_path: Path) -> CRS | None:
    return _load_crs_from_json(json_file_path, "dsm_")


def _load_crs_from_json(json_file_path: Path, prefix: str) -> CRS | None:
    """
    JSONファイルから座標系情報を読み取ります。

    :param json_file_path: JSONファイルのパス
    :type json_file_path: Path
    :param prefix: プレフィックス文字列
    :type prefix: str
    :returns: 座標系文字列（見つからない場合はNone）
    :rtype: CRS | None
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        crs_str = json_data.get(f"{prefix}crs")
        if crs_str:
            return CRS.from_string(crs_str)
    return None


def update_json(json_file_path: Path, data: dict):
    """
    JSONファイルに書き込む。

    :param json_file_path: JSONファイルのパス
    :type json_file_path: Path
    :param data: 更新するデータ
    :type data: dict
    """
    # 既存のデータを読み込み
    if json_file_path.exists():
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                old = json.load(f)
                old.update(data)
                data = old
        except JSONDecodeError as e:
            print(
                f"{json_file_path}をJSONとして読み込めなかったため、全体を上書きします: {e}",
                file=sys.stderr,
            )
            json_file_path.unlink()

    json_file_path.parent.mkdir(parents=True, exist_ok=True)

    # ファイルに書き戻し
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
