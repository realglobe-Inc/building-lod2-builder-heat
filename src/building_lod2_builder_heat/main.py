import json
import math
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
    canvas_size = model.load_checkpoint(str(checkpoint_file))
    if canvas_size is None:
        canvas_size = 256

    dsm_files = dsm_dir.glob("*.las")
    for dsm_file in dsm_files:
        print(f"{dsm_file}を処理します")
        dsm_bgr, dsm_depth, dsm_left, dsm_top, dsm_right, dsm_bottom, dsm_crs = (
            _load_las(dsm_file)
        )

        # 入力画像を出力する
        if intermediate_dir:
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(intermediate_dir / f"{dsm_file.stem}_dsm_rgb.png"), dsm_bgr)
            cv2.imwrite(
                str(intermediate_dir / f"{dsm_file.stem}_dsm_depth.png"), dsm_depth
            )

        ortho_file: Path | None = (
            (ortho_dir / dsm_file.name).with_suffix(".tif") if ortho_dir else None
        )
        # if ortho_file:
        #     ortho_bgr, ortho_bounds, ortho_crs = _load_ortho(ortho_file)
        # TODO dsmとorthoの混合

        input_bgr = dsm_bgr
        padded_bgr, left_idx, top_idx, right_idx, bottom_idx = _pad_input_image(
            input_bgr, canvas_size
        )
        # TODO depthの利用

        # 入力画像を出力する
        if intermediate_dir:
            cv2.imwrite(
                str(intermediate_dir / f"{dsm_file.stem}_padded_rgb.png"), padded_bgr
            )

        corners, edges = model.infer(padded_bgr)

        print(f"検出結果: {len(corners)}個のコーナー、{len(edges)}個のエッジ")
        _update_json(
            output_dir / f"{dsm_file.stem}.json",
            {
                "corners": corners.tolist(),
                "edges": edges.tolist(),
                "canvas_size": canvas_size,
                "image_left_index": left_idx,
                "image_top_index": top_idx,
                "image_right_index": right_idx,
                "image_bottom_index": bottom_idx,
                "dsm_left_bound": dsm_left,
                "dsm_top_bound": dsm_top,
                "dsm_right_bound": dsm_right,
                "dsm_bottom_bound": dsm_bottom,
                "dsm_crs": dsm_crs.to_string() if dsm_crs else None,
            },
        )

        if intermediate_dir:
            visualized_bgr = padded_bgr.copy()
            for edge in edges:
                cv2.line(
                    visualized_bgr,
                    corners[edge[0]],
                    corners[edge[1]],
                    (0, 255, 0),
                    1,
                )
            cv2.imwrite(
                str(intermediate_dir / f"{dsm_file.stem}_edges.png"), visualized_bgr
            )

        pass


def _pad_input_image(
    input_bgr: NDArray[np.uint8], canvas_size: int
) -> tuple[NDArray[np.uint8], int, int, int, int]:
    """
    カラー画像を正方形のキャンバスに収める。

    画像の幅または高さがキャンバスの辺を超える場合、大きい方の寸法が辺と同じになるまでアスペクト比を維持して画像を縮小する。
    画像はキャンバスの中央に配置する。

    input_bgr: 入力画像
    NDArray[np.uint8]: NDArray[np.uint8]
    return: 戻り値
    tuple[NDArray[np.uint8], int, int, int, int]: 画像を収めたキャンバスとキャンバス内での画像の左、上、右、下のインデクス
    """
    result = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

    # 長辺がcanvas_sizeになるようにリサイズ
    input_bgr = _fit_image(input_bgr, canvas_size)
    image_height, image_width = input_bgr.shape[:2]

    # 中央寄せでコピー
    left = (canvas_size - image_width) // 2
    top = (canvas_size - image_height) // 2
    right = left + image_width
    bottom = top + image_height
    result[top : top + image_height, left : left + image_width] = input_bgr

    return result, left, top, right, bottom


def _load_las(
    las_file_path: Path,
) -> tuple[
    NDArray[np.uint8], NDArray[np.uint8], float, float, float, float, CRS | None
]:
    """
    与えられたファイルパスからLASデータを読み込み、RGB、深度、座標範囲、およびCRS（座標参照系）情報を返します。

    この関数は、3D LAS点群データを2D格子構造に変換し、各格子点にRGBと深度の値を割り当てます。
    RGBと深度の両方の表現について、欠損格子を補間することがあります。

    :param las_file_path: 読み込んで処理するLASファイルへのパス。
    :type las_file_path: Path
    :return: 次を含むタプル：
        - RGB画像を表すNumPy配列
        - 深度画像を表すNumPy配列
        - 地図上の左端
        - 地図上の上端
        - 地図上の右端
        - 地図上の下端
        - LASヘッダーまたは関連ファイルから抽出されたCRS（座標参照系）。
    :rtype: tuple[NDArray[np.uint8], NDArray[np.uint8], float, float, float, float, CRS | None]
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

    bgr_data = np.full((len(x_values), len(y_values), 3), 255, dtype=np.uint8)
    depth_data = np.full((len(x_values), len(y_values)), 255, dtype=np.uint8)
    inpaint_mask = np.zeros((len(x_values), len(y_values)), dtype=np.bool)
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            idx = int(grid_indices[i, j])
            if idx < 0:
                inpaint_mask[i, j] = True
            else:
                # OpenCVはBGR
                bgr_data[i, j, 0] = las_data.blue[idx]
                bgr_data[i, j, 1] = las_data.green[idx]
                bgr_data[i, j, 2] = las_data.red[idx]
                depth_data[i, j] = int(
                    math.floor(((las_data.z[idx] - z_min) / z_range) * 256)
                )

    # bgr_dataとdepth_dataの穴あき部分を補完する
    # 1マス離れた箇所にinpaint_maskがFalseの箇所があれば、その平均にする。
    # 1マス離れた箇所に無ければ、2マス離れた場所を調べ、inpaint_maskがFalseの箇所があれば、その平均する
    # 2マス離れた場所にもなければ、そのまま

    height, width = inpaint_mask.shape
    a = 0
    b = 0
    c = 0
    for i in range(height):
        for j in range(width):
            if not inpaint_mask[i, j]:
                b += 1
                continue
            # 1マス離れた近傍を探索
            nearby = []
            for di, dj in [
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1),
            ]:
                ni = i + di
                nj = j + dj
                if 0 <= ni < height and 0 <= nj < width and not inpaint_mask[ni, nj]:
                    nearby.append((ni, nj))
            if len(nearby) == 0:
                # 2マス離れた近傍を探索
                for di, dj in [
                    (2, 0),
                    (1, 1),
                    (0, 2),
                    (-1, 1),
                    (-2, 0),
                    (-1, -1),
                    (0, -2),
                    (1, -1),
                ]:
                    ni = i + di
                    nj = j + dj
                    if (
                        0 <= ni < height
                        and 0 <= nj < width
                        and not inpaint_mask[ni, nj]
                    ):
                        nearby.append((ni, nj))

            if nearby:
                a += 1
                # 2マス離れた点の平均をとる
                bgr_data[i, j] = np.mean(
                    [bgr_data[ni, nj] for ni, nj in nearby], axis=0
                ).astype(np.uint8)
                depth_data[i, j] = int(
                    np.mean([depth_data[ni, nj] for ni, nj in nearby])
                )
            else:
                c += 1

    if crs is not None and crs.axis_info[0].direction != "east":
        bgr_data = bgr_data.transpose(1, 0, 2)
        depth_data = depth_data.T

    return (
        bgr_data,
        depth_data,
        float(x_values[0]),
        float(y_values[0]),
        float(x_values[-1]),
        float(y_values[-1]),
        crs,
    )


def _points_to_grid(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    点群を格子状に構成します。
    入力点群から一意のX座標とY座標を列挙して格子を切ります。
    (X座標, Y座標)が一致する点が入力点群に存在する場合はその点が格子点に配置されます。
    存在しない場合は格子点はNoneになります。

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

    # 格子の切れ目の位置
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

    # 格子を作成する
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


def _fit_image(
    image: NDArray[np.uint8], size: int, max_scale_factor: float = 4.0
) -> NDArray[np.uint8]:
    """
    補間法を用いて、与えられた画像の長辺を目的のサイズに合わせます。

    この関数は、NumPy配列で表された入力画像を受け取り、補間技術を用いて指定されたサイズに拡大縮小します。
    この関数は、画像が画像処理タスクで一般的な8ビット符号なし整数配列の形式であることを前提としています。

    :param image: 入力画像。
    :type image: NDArray[np.uint8]
    :param size: 長辺を合わせるサイズ。
    :type size: int
    :param max_scale_factor: 最大拡大率。
    :type max_scale_factor: float
    :return: 拡大縮小された画像。
    :rtype: NDArray[np.uint8]
    """
    current_height, current_width = image.shape[:2]
    if current_height > current_width:
        height = size
        width = int(round(size * current_width / current_height))
    else:
        height = int(round(size * current_height / current_width))
        width = size
    if current_height == height:
        return image
    elif current_height > height:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)

    scale_factor = height / current_height
    if scale_factor > max_scale_factor:
        width = int(round(max_scale_factor * current_width))
        height = int(round(max_scale_factor * current_height))

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)


if __name__ == "__main__":
    app()
