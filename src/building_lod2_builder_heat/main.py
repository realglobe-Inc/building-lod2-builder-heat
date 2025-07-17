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
        None, "--ortho-dir", help="オルソファイルを読み取るディレクトリ。", exists=True
    ),
    obj_dir: Path | None = typer.Option(
        None, "--obj-dir", help="OBJファイルを読み取るディレクトリ。", exists=True
    ),
    intermediate_dir: Path | None = typer.Option(
        None, "--intermediate-dir", help="中間生成物を保存するディレクトリ。"
    ),
    gpu: bool = typer.Option(True, "--gpu/--cpu", help="GPUで実行するかどうか"),
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
        print(f"{dsm_file}を処理します")

        outline: GeoOutline | None = None
        if obj_dir is not None:
            obj_file = (obj_dir / dsm_file.name).with_suffix(".obj")
            outline = load_outline_from_obj(obj_file)

        ortho_bgr: NDArray[np.uint8] | None = None
        ortho_bounds: Bounds | None = None
        if ortho_dir:
            ortho_file = (ortho_dir / dsm_file.name).with_suffix(".tif")
            ortho_bgr, ortho_bounds = load_ortho(
                ortho_file,
                canvas_size=(canvas_size, canvas_size),
                outline=outline,
            )

        dsm_bgr, dsm_depth, dsm_bounds = load_las(
            dsm_file,
            canvas_size=(
                (canvas_size, canvas_size)
                if ortho_bgr is None
                else tuple(ortho_bgr.shape[:2][:, -1])
            ),
            outline=outline,
            canvas_bounds=ortho_bounds,
        )

        # 入力画像を出力する
        if intermediate_dir:
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(intermediate_dir / f"{dsm_file.stem}_dsm_rgb.png"), dsm_bgr)
            cv2.imwrite(
                str(intermediate_dir / f"{dsm_file.stem}_dsm_depth.png"), dsm_depth
            )

        input_bgr = ortho_bgr if ortho_bgr is not None else dsm_bgr
        padded_bgr, bounds = _pad_image(input_bgr, canvas_size)
        padded_depth, _ = _pad_image(dsm_depth, canvas_size, pixel_size=1)

        # 入力画像を出力する
        if intermediate_dir:
            cv2.imwrite(
                str(intermediate_dir / f"{dsm_file.stem}_padded_rgb.png"), padded_bgr
            )
            cv2.imwrite(
                str(intermediate_dir / f"{dsm_file.stem}_padded_depth.png"),
                padded_depth,
            )

        # TODO depthの利用
        corners, edges = model.infer(padded_bgr)

        print(f"検出結果: {len(corners)}個のコーナー、{len(edges)}個のエッジ")
        result_data = {
            "corners": corners.tolist(),
            "edges": edges.tolist(),
            "canvas_size": canvas_size,
            "image_bounds": bounds.ltrb,
        }
        if dsm_bounds:
            result_data["geo_bounds"] = dsm_bounds.ltrb
            result_data["geo_crs"] = dsm_bounds.crs.to_string()
        update_json(output_dir / f"{dsm_file.stem}.json", result_data)

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
            for corner in corners:
                cv2.circle(
                    visualized_bgr,
                    corner,
                    2,
                    (255, 0, 0),
                    -1,
                )
            cv2.imwrite(
                str(intermediate_dir / f"{dsm_file.stem}_edges.png"), visualized_bgr
            )

        pass


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
