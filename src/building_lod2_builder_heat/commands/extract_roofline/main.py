import os
import sys
import traceback
from pathlib import Path

import typer
from heat import HEAT

from building_lod2_builder_heat.commands.extract_roofline.main_unit import main_unit
from building_lod2_builder_heat.common import file_names, parameter_keys
from building_lod2_builder_heat.common.parameter import (
    load_parameter,
    update_parameters,
)

app = typer.Typer()


@app.command()
def run(
    checkpoint_file_path: Path = typer.Argument(
        help="学習済みモデルのパス。", exists=True
    ),
    data_root_dir_path: Path = typer.Argument(
        help=f"データディレクトリのパス。各サブディレクトリの以下のファイルが使われます。\n必須: {file_names.ROOFLINE_EXTRACTION_INPUT_RGB}, {file_names.ROOFLINE_EXTRACTION_INPUT_DEPTH}。\nオプション: なし。",
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

    print(f"モデルをロードします: {checkpoint_file_path}")
    model = HEAT(force_cpu=not prefer_gpu)
    model.load_checkpoint(checkpoint_file_path)

    output_root_dir_path = (
        output_root_dir_path if output_root_dir_path is not None else data_root_dir_path
    )

    input_dir_paths = [p for p in data_root_dir_path.iterdir() if p.is_dir()]
    for input_dir_path in sorted(input_dir_paths):
        target_id = input_dir_path.stem

        rgb_file_path = input_dir_path / file_names.ROOFLINE_EXTRACTION_INPUT_RGB
        depth_file_path = input_dir_path / file_names.ROOFLINE_EXTRACTION_INPUT_DEPTH
        if not rgb_file_path.exists() or not depth_file_path.exists():
            print(f"データが足りないため{target_id}をスキップします")
            continue

        output_dir_path = output_root_dir_path / target_id
        output_file_path = output_dir_path / file_names.EXTRACT_ROOFLINE_OUTPUT
        if (
            skip_exist
            and load_parameter(output_file_path, parameter_keys.ROOFLINE_EDGES)
            is not None
        ):
            print(f"{target_id}をスキップします")
            continue

        print(f"{target_id}を処理します")
        output_dir_path.mkdir(parents=True, exist_ok=True)

        byproduct_dir_path: Path | None = None
        if byproduct_root_dir_path is not None:
            byproduct_dir_path = byproduct_root_dir_path / target_id
            byproduct_dir_path.mkdir(parents=True, exist_ok=True)

        try:
            main_unit(
                rgb_file_path=rgb_file_path,
                depth_file_path=depth_file_path,
                model=model,
                output_dir_path=output_dir_path,
                byproduct_dir_path=byproduct_dir_path,
            )
        except Exception as e:
            print(f"{target_id}の処理に失敗しました", file=sys.stderr)
            if exit_on_error:
                raise
            tb = traceback.format_exc()
            traceback.print_exc()
            update_parameters(
                output_dir_path / file_names.EXTRACT_ROOFLINE_OUTPUT,
                {parameter_keys.ERROR: str(e), parameter_keys.TRACEBACK: tb},
            )
            continue


if __name__ == "__main__":
    app()
