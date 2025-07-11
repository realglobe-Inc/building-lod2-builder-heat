import json
import sys
from json import JSONDecodeError
from pathlib import Path

import cv2
import laspy
import numpy as np
import typer
from heat import HEAT
from laspy import LasData
from numpy.typing import NDArray
from pyproj import CRS

# os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"


app = typer.Typer()


@app.command()
def run(
    checkpoint_file: Path = typer.Argument(help="学習済みモデルのパス。", exists=True),
    dsm_dir: Path = typer.Argument(
        help="DSMファイルを読み取るディレクトリ。", exists=True
    ),
    output_dir: Path = typer.Argument(
        help="抽出した屋根線上方を出力するディレクトリのパス。"
    ),
    ortho_dir: Path | None = typer.Option(
        None, "--ortho-dir", help="オルソファイルを読み取るディレクトリ。", exists=True
    ),
    intermediate_dir: Path | None = typer.Option(
        None, "--intermediate-dir", help="中間生成物を保存するディレクトリ。"
    ),
    gpu: bool = typer.Option(True, "--gpu/--cpu", help="GPUで実行するかどうか"),
):
    print(f"モデルをロードします: {checkpoint_file}")
    model = HEAT(gpu)
    model.load_checkpoint(str(checkpoint_file))

    dsm_files = dsm_dir.glob("*.las")
    for dsm_file in dsm_files:
        print(f"{dsm_file}を処理します")
        dsm_bgr, dsm_depth, dsm_bounds, dsm_crs = _load_las(dsm_file)

        # 中間ファイルを出力する
        if intermediate_dir:
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(intermediate_dir / f"{dsm_file.stem}_dsm_rgb.png"), dsm_bgr)
            cv2.imwrite(
                str(intermediate_dir / f"{dsm_file.stem}_dsm_depth.png"), dsm_depth
            )
            _update_json(
                intermediate_dir / f"{dsm_file.stem}_param.json",
                {"dsm_bounds": list(dsm_bounds)},
            )

        ortho_file: Path | None = (
            (ortho_dir / dsm_file.name).with_suffix(".tif") if ortho_dir else None
        )
        # if ortho_file:
        #     ortho_bgr, ortho_bounds, ortho_crs = _load_ortho(ortho_file)

        input_bgr = dsm_bgr
        input_bgr_256 = _resize_input_image(input_bgr)

        corners, edges = model.infer(input_bgr_256)

        print(f"検出結果: {len(corners)}個のコーナー、{len(edges)}個のエッジ")
        _update_json(
            output_dir / f"{dsm_file.stem}.json",
            {
                "corners": corners.tolist(),
                "edges": edges.tolist(),
            },
        )

        pass


def _resize_input_image(input_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    result = np.zeros((256, 256, 3), dtype=np.uint8)
    h, w = input_bgr.shape[:2]
    if h <= 256 and w <= 256:
        # 256x256以下の場合は中央寄せでコピー
        x_offset = (256 - w) // 2
        y_offset = (256 - h) // 2
        result[y_offset : y_offset + h, x_offset : x_offset + w] = input_bgr
    else:
        # 256を超える場合は長辺が256になるようにリサイズ
        if h > w:
            new_h = 256
            new_w = int(w * 256 / h)
        else:
            new_h = int(h * 256 / w)
            new_w = 256
        resized = cv2.resize(input_bgr, (new_w, new_h))
        x_offset = (256 - new_w) // 2
        y_offset = (256 - new_h) // 2
        result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return result


def _load_las(
    las_file_path: Path,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.float32], CRS | None]:
    """
    与えられたファイルパスからLASデータを読み込み、RGB、深度、座標範囲データ、およびCRS（座標参照系）情報を生成します。

    この関数は、3D LAS点群データを2Dグリッド構造に変換し、各グリッドセルにRGBと深度の値を割り当て、
    RGBと深度の両方の表現について欠損グリッドデータを補間することを含みます。

    :param las_file_path: 読み込んで処理するLASファイルへのパス。
    :type las_file_path: Path
    :return: 次を含むタプル：
        - 2DグリッドのRGBデータを表すnp.uint8型の3D NumPy配列。
        - 2Dグリッドの深度値を表すnp.uint8型の2D NumPy配列。
        - グリッドの空間範囲（x_min、y_min、x_max、y_max）を定義する1D NumPy配列。
        - LASヘッダーまたは関連ファイルから抽出されたCRS（座標参照系）。
    :rtype: tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32], CRS | None]
    """
    las_data: LasData
    with laspy.open(las_file_path) as f:
        las_data = f.read()

    crs: CRS | None = las_data.header.parse_crs()
    if crs is None:
        crs = _load_crs(las_file_path.with_suffix(".json"))

    points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()
    grid_indices, x_values, y_values = _points_to_grid(points)
    z_min = points[:, 2].min()
    z_range = points[:, 2].max() - z_min
    if z_range < 10:
        # 高低差10mは入るようにする。最高最低を範囲にすると、ほんの数cmで黒から白に変わっちゃったりする
        z_min = z_min - (10 - z_range) / 2
        z_range = 10

    bgr_data = np.zeros((len(x_values), len(y_values), 3), dtype=np.uint8)
    depth_data = np.zeros((len(x_values), len(y_values)), dtype=np.uint8)
    inpaint_mask = np.zeros((len(x_values), len(y_values)), dtype=np.uint8)
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            if grid_indices[i, j] >= 0:
                idx = int(grid_indices[i, j])
                if idx:
                    # OpenCVはBGR
                    bgr_data[i, j, 0] = las_data.blue[idx]
                    bgr_data[i, j, 1] = las_data.green[idx]
                    bgr_data[i, j, 2] = las_data.red[idx]
                    depth_data[i, j] = int(
                        round(((las_data.z[idx] - z_min) / z_range) * 255)
                    )
                else:
                    inpaint_mask[i, j] = 255

    # bgr_dataとdepth_dataの穴あき部分を補完する
    bgr_data = cv2.inpaint(bgr_data, inpaint_mask, 3, cv2.INPAINT_TELEA)
    depth_data = cv2.inpaint(depth_data, inpaint_mask, 3, cv2.INPAINT_TELEA)

    if crs is not None and crs.axis_info[0].direction != "east":
        bgr_data = bgr_data.transpose(1, 0, 2)
        depth_data = depth_data.T

    bounds = np.array([x_values[0], y_values[-1], x_values[-1], y_values[0]])
    return bgr_data, depth_data, bounds, crs


def _points_to_grid(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    点を2次元構造グリッドに変換します。
    入力点群から一意のX座標とY座標を列挙してグリッドを作成します。
    (X座標, Y座標)が一致する点が入力点群に存在する場合はその点がセルに配置されます。
    存在しない場合はセルはNoneになります。

    points: 3つの値（x座標とy座標とz座標）を持つ点を各行で表すnumpy配列。
    return: 行と列がそれぞれ入力点群のどの位置かを示す2次元numpy配列。
        ソートされたx値の列。
        ソートされたy値の列。
    """
    # X座標とY座標の一意な値を取得
    unique_x = np.unique(points[:, 0])
    unique_y = np.unique(points[:, 1])

    def fill_value(values):
        # 最小間隔を計算
        diffs = np.diff(values)
        min_diff = np.min(diffs[diffs > 0])

        # 補完
        new_values = [values[0]]
        for i in range(1, len(values)):
            diff = values[i] - values[i - 1]
            if diff > min_diff:
                steps = int(round(diff / min_diff))
                for j in range(1, steps):
                    new_values.append(values[i - 1] + j * min_diff)
            new_values.append(values[i])
        return np.array(new_values)

    # グリッドセル
    x_sorted = fill_value(np.sort(unique_x))
    # 配列の先頭が画像の先頭になるように
    y_sorted = fill_value(np.sort(unique_y))[::-1]

    # 参照辞書を作成する
    x_to_index = {}
    for i, x in enumerate(x_sorted):
        x_to_index[x] = i
    y_to_index = {}
    for i, y in enumerate(y_sorted):
        y_to_index[y] = i

    # グリッドを作成します
    grid_indices = np.empty((len(x_sorted), len(y_sorted)), dtype=int)
    grid_indices.fill(-1)

    for i, point in enumerate(points):
        x_index = x_to_index[point[0]]
        y_index = y_to_index[point[1]]
        grid_indices[x_index, y_index] = i

    return grid_indices, x_sorted, y_sorted


def _load_crs(json_file_path: Path) -> CRS | None:
    """
    JSONファイルから座標系情報を読み取ります。

    :param json_file_path: JSONファイルのパス
    :return: 座標系文字列（見つからない場合はNone）
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        crs_str = json_data.get("obj_crs")
        if crs_str:
            return CRS.from_string(crs_str)
    return None


def _update_json(json_file_path: Path, data: dict):
    """
    JSONファイルに書き込む。

    :param json_file_path: JSONファイルのパス
    :param data: 更新するデータ
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
                f"{json_file_path}をJSONとして読み込めなかったため、全体を上書きします",
                file=sys.stderr,
            )
            json_file_path.unlink()

    json_file_path.parent.mkdir(parents=True, exist_ok=True)

    # ファイルに書き戻し
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    app()
