import os
import sys
import traceback
from pathlib import Path

import typer
from heat import HEAT

from building_lod2_builder_heat.commands.extract_roofline import (
    file_names,
    parameter_keys,
)
from building_lod2_builder_heat.commands.extract_roofline.main_unit import main_unit
from building_lod2_builder_heat.commands.extract_roofline.parameter import (
    load_parameter,
)

app = typer.Typer()


@app.command()
def run(
    checkpoint_file: Path = typer.Argument(help="学習済みモデルのパス。", exists=True),
    data_root_dir_path: Path = typer.Argument(
        help=f"データディレクトリのパス。各サブディレクトリの以下のファイルが使われます。\n必須: {file_names.CLIPPED_DSM}。\nオプション: {file_names.PARAMETERS}, {file_names.LOD1_OBJ}, {file_names.CLIPPED_ORTHO}。",
        exists=True,
    ),
    output_root_dir_path: Path | None = typer.Option(
        None,
        "--output-dir",
        help="出力ディレクトリのパス。無指定の場合はデータディレクトリに出力する。",
    ),
    byproduct_root_dir_path: Path | None = typer.Option(
        None, "--byproduct-dir", help="副産物を保存するディレクトリ。"
    ),
    prefer_gpu: bool = typer.Option(
        True,
        "--prefer-gpu/--force-cpu",
        help="GPUが利用可能ならGPUを利用するかどうか。",
    ),
    skip_exist: bool = typer.Option(
        True,
        "--skip-exist/--overwrite",
        help="既に結果が存在する場合はスキップするかどうか。",
    ),
    rich_error: bool = typer.Option(
        True,
        "--rich-error/--normal-error",
        help="エラー時に変数の内容等まで出力するか。",
    ),
    exit_on_error: bool = typer.Option(
        False,
        "--exit-on-error",
        help="1つの処理対象に対するエラーで終了する。",
    ),
):
    """
    屋根線を抽出する。
    """
    os.environ["_TYPER_STANDARD_TRACEBACK"] = "" if rich_error else "true"

    print(f"モデルをロードします: {checkpoint_file}")
    model = HEAT(force_cpu=not prefer_gpu)
    canvas_size = model.load_checkpoint(checkpoint_file)
    if canvas_size is None:
        canvas_size = 256

    output_root_dir_path = (
        output_root_dir_path if output_root_dir_path is not None else data_root_dir_path
    )

    input_dir_paths = [p for p in data_root_dir_path.iterdir() if p.is_dir()]
    for input_dir_path in sorted(input_dir_paths):
        target_id = input_dir_path.stem

        output_dir_path = output_root_dir_path / target_id
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir_path / file_names.PARAMETERS
        if (
            skip_exist
            and load_parameter(output_file_path, parameter_keys.ROOFLINE_EDGES)
            is not None
        ):
            print(f"{target_id}をスキップします")
            continue

        dsm_file_path = input_dir_path / file_names.CLIPPED_DSM
        if not dsm_file_path.exists():
            print(f"入力データが足りないため{target_id}をスキップします")
            continue

        param_file_path = input_dir_path / file_names.PARAMETERS
        if not param_file_path.exists():
            param_file_path = None

        obj_file_path = input_dir_path / file_names.LOD1_OBJ
        if not obj_file_path.exists():
            obj_file_path = None

        ortho_file_path = input_dir_path / file_names.CLIPPED_ORTHO
        if not ortho_file_path.exists():
            ortho_file_path = None

        byproduct_dir_path = None
        if byproduct_root_dir_path is not None:
            byproduct_dir_path = byproduct_root_dir_path / target_id
            byproduct_dir_path.mkdir(parents=True, exist_ok=True)

        print(f"{target_id}を処理します")
        try:
            main_unit(
                model=model,
                dsm_file_path=dsm_file_path,
                canvas_size=canvas_size,
                output_dir_path=output_dir_path,
                param_file_path=param_file_path,
                obj_file_path=obj_file_path,
                ortho_file_path=ortho_file_path,
                byproduct_dir_path=byproduct_dir_path,
            )
        except Exception:
            print(f"{target_id}の処理に失敗しました", file=sys.stderr)
            if exit_on_error:
                raise
            traceback.print_exc()
            continue


if __name__ == "__main__":
    app()
