import json
import sys
from json import JSONDecodeError
from pathlib import Path

from pyproj import CRS

from building_lod2_builder_heat.commands.extract_roofline import parameter_keys


def load_crs_parameter(param_file_path: Path) -> CRS | None:
    """
    パラメータファイルから座標系情報を読み取る。

    :param param_file_path: パラメータファイルのパス
    :return: 座標系文字列（見つからない場合はNone）
    """
    crs_str = load_parameter(param_file_path, parameter_keys.CRS)
    if crs_str is None or not isinstance(crs_str, str):
        return None
    return CRS.from_string(crs_str)


def load_parameter(param_file_path: Path, key: str) -> object:
    try:
        with open(param_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            return json_data.get(key)
    except (FileNotFoundError, JSONDecodeError):
        return None


def update_crs_parameter(param_file_path: Path, crs: CRS):
    """
    パラメータファイルに座標系を書き込む。

    :param param_file_path: パラメータファイルのパス
    :param crs: 座標参照系情報
    """
    update_parameters(param_file_path, {parameter_keys.CRS: crs.to_string()})


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
            print(
                f"{param_file_path}をJSONとして読み込めなかったため、全体を上書きします: {e}",
                file=sys.stderr,
            )
            param_file_path.unlink()

    # ファイルに書き戻し
    param_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(param_file_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
