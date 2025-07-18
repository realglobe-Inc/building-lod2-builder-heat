import json
import sys
from pathlib import Path

import cv2
import numpy as np
import typer
from heat import HEAT
from numpy.typing import NDArray

from building_lod2_builder_heat.bounds import Bounds
from building_lod2_builder_heat.las import load_las
from building_lod2_builder_heat.obj import load_outline_from_obj
from building_lod2_builder_heat.ortho import load_ortho
from building_lod2_builder_heat.outline import GeoOutline
from building_lod2_builder_heat.parameter import update_json

# os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"


app = typer.Typer()


@app.command()
def run(
    checkpoint_file: Path = typer.Argument(help="学習済みモデルのパス。", exists=True),
    dsm_dir: Path = typer.Argument(
        help="DSMファイルを読み取るディレクトリ。", exists=True
    ),
    output_dir: Path = typer.Argument(
        help="抽出した屋根線情報を出力するディレクトリのパス。"
    ),
    ortho_dir: Path | None = typer.Option(
        None,
        "--ortho-dir",
        help="オルソファイルを読み取るディレクトリ。\n拡張子以外DSMと同じファイル名のファイルがある場合、DSMの代わりにRGB画像を取得する。",
        exists=True,
    ),
    obj_dir: Path | None = typer.Option(
        None,
        "--obj-dir",
        help="OBJファイルを読み取るディレクトリ。\n拡張子以外DSMと同じファイル名のファイルがある場合、外形線を画像拡大時の縁の処理に利用する。",
        exists=True,
    ),
    intermediate_dir: Path | None = typer.Option(
        None, "--intermediate-dir", help="中間生成物を保存するディレクトリ。"
    ),
    gpu: bool = typer.Option(True, "--gpu/--cpu", help="GPUで実行するかどうか"),
    skip_exist: bool = typer.Option(
        True,
        "--skip-exist/--overwrite",
        help="既に結果が存在する場合はスキップするかどうか。",
    ),
):
    """
    屋根線を抽出する。
    """

    print(f"モデルをロードします: {checkpoint_file}")
    model = HEAT(gpu)
    canvas_size = model.load_checkpoint(str(checkpoint_file))
    if canvas_size is None:
        canvas_size = 256

    dsm_files = dsm_dir.glob("*.las")
    for dsm_file in dsm_files:
        output_file = output_dir / f"{dsm_file.stem}.json"
        if skip_exist and output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                if "roof_corners" in json_data:
                    print(f"{dsm_file}をスキップします")
                    continue

        files_dir = None
        if intermediate_dir is not None:
            files_dir = intermediate_dir / dsm_file.stem
            files_dir.mkdir(parents=True, exist_ok=True)

        print(f"{dsm_file}を処理します")

        outline: GeoOutline | None = None
        if obj_dir is not None:
            obj_file = (obj_dir / dsm_file.name).with_suffix(".obj")
            outline = load_outline_from_obj(obj_file)
            if outline is not None:
                print(f"{obj_file}から外形線を読み込みました")

        ortho_bgr: NDArray[np.uint8] | None = None
        ortho_bounds: Bounds | None = None
        if ortho_dir:
            ortho_file = (ortho_dir / dsm_file.name).with_suffix(".tif")
            ortho_bgr, ortho_bounds = load_ortho(
                ortho_file,
                canvas_size=(canvas_size, canvas_size),
                outline=outline,
            )

            if ortho_bgr is not None:
                print(f"{ortho_file}からRGB画像を読み込みました")

                # 入力画像を出力する
                if files_dir is not None:
                    cv2.imwrite(str(files_dir / "ortho.png"), ortho_bgr)

        dsm_bgr, dsm_depth, dsm_bounds = load_las(
            dsm_file,
            canvas_size=(
                (canvas_size, canvas_size)
                if ortho_bgr is None
                else (ortho_bgr.shape[1], ortho_bgr.shape[0])
            ),
            outline=outline,
            canvas_bounds=ortho_bounds,
        )
        if dsm_depth is None:
            print(f"{dsm_file}を読み込めませんでした", file=sys.stderr)
            continue

        # 入力画像を出力する
        if files_dir:
            cv2.imwrite(str(files_dir / "dsm_rgb.png"), dsm_bgr)
            cv2.imwrite(str(files_dir / "dsm_depth.png"), dsm_depth)

        input_bgr = ortho_bgr if ortho_bgr is not None else dsm_bgr
        padded_bgr, bounds = _pad_image(input_bgr, canvas_size)
        padded_depth, _ = _pad_image(dsm_depth, canvas_size, pixel_size=1)

        # 入力画像を出力する
        if files_dir:
            cv2.imwrite(str(files_dir / "padded_rgb.png"), padded_bgr)
            cv2.imwrite(
                str(files_dir / "padded_depth.png"),
                padded_depth,
            )

        # TODO depthの利用
        corners, edges = model.infer(padded_bgr)

        print(f"検出結果: {len(corners)}個の角 {len(edges)}個の辺")
        result_data = {
            "roof_corners": corners.tolist(),
            "roof_edges": edges.tolist(),
            "roof_canvas_size": canvas_size,
            "roof_image_bounds": bounds.ltrb,
            "roof_geo_bounds": dsm_bounds.ltrb,
            "roof_crs": dsm_bounds.crs.to_string(),
        }
        update_json(output_file, result_data)

        # 結果画像を出力する
        if files_dir:
            visualized_bgr = padded_bgr.copy()
            for edge in edges:
                cv2.line(
                    visualized_bgr,
                    corners[edge[0]],
                    corners[edge[1]],
                    (0, 255, 0),
                    1,
                )
            for corner in corners:
                cv2.circle(
                    visualized_bgr,
                    corner,
                    2,
                    (255, 0, 0),
                    -1,
                )
            cv2.imwrite(str(files_dir / "result.png"), visualized_bgr)


def _pad_image(
    input_image: NDArray[np.uint8],
    canvas_size: int,
    pixel_size: int = 3,
    pad_value: int = 255,
) -> tuple[NDArray[np.uint8], Bounds]:
    """
    画像を正方形のキャンバスに収める。

    input_image: 入力画像
    return: キャンバスの中央に配置された画像とその配置場所
    """

    image_height, image_width = input_image.shape[:2]

    # 中央寄せでコピー
    left = (canvas_size - image_width) // 2
    top = (canvas_size - image_height) // 2
    right = left + image_width
    bottom = top + image_height

    shape = (
        (canvas_size, canvas_size, pixel_size)
        if pixel_size > 1
        else (canvas_size, canvas_size)
    )
    canvas = np.full(shape, pad_value, dtype=np.uint8)
    canvas[top:bottom, left:right] = input_image
    bounds = Bounds(left, top, right, bottom)

    return canvas, bounds


if __name__ == "__main__":
    app()
