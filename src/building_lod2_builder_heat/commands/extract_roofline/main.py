import os
import traceback
from pathlib import Path

import typer
from heat import HEAT
from loguru import logger
from torch.utils.data import DataLoader

from building_lod2_builder_heat.commands.extract_roofline.dataset import (
    RooflineDataset,
)
from building_lod2_builder_heat.commands.extract_roofline.main_unit import (
    save_roofline_result,
)
from building_lod2_builder_heat.common import file_names, parameter_keys
from building_lod2_builder_heat.common.logging import LogLevel, setup_logger
from building_lod2_builder_heat.common.parameter import (
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
    backup: bool = typer.Option(
        False,
        "--backup",
        help="出力ファイルが既に存在する場合にバックアップを作成する。",
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
    log_level: LogLevel = typer.Option(
        LogLevel.INFO,
        "--log-level",
        help="ログレベルを指定します。",
        case_sensitive=False,
    ),
    log_file: Path | None = typer.Option(
        None,
        "--log-file",
        help="ログファイルの出力先パス。",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        help="一度に推論するサンプル数。",
        min=1,
    ),
    num_workers: int = typer.Option(
        0,
        "--num-workers",
        help="DataLoaderで使用するワーカー数。",
        min=0,
    ),
):
    """
    屋根線を抽出する。
    """
    setup_logger(log_level, log_file)
    os.environ["_TYPER_STANDARD_TRACEBACK"] = "" if rich_error else "true"

    logger.info(f"{checkpoint_file_path} をロードします")
    model = HEAT(force_cpu=not prefer_gpu)
    model.load_checkpoint(checkpoint_file_path)

    logger.debug(f"演算デバイス: {model.device}")

    output_root_dir_path = (
        output_root_dir_path if output_root_dir_path is not None else data_root_dir_path
    )

    dataset = RooflineDataset(
        data_root_dir_path=data_root_dir_path,
        output_root_dir_path=output_root_dir_path,
        skip_exist=skip_exist,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda x: x,
    )

    for batch in dataloader:
        # 画像を抽出してバッチ推論
        bgr_images = [sample["bgr_image"] for sample in batch]
        try:
            batch_results = model.infer_batch(bgr_images)
        except Exception as e:
            logger.error(f"バッチ推論中にエラーが発生しました: {e}")
            if exit_on_error:
                raise
            continue

        # 各サンプルの結果を保存
        for sample, (corners, edges) in zip(batch, batch_results):
            building_id = sample["building_id"]
            output_dir_path = sample["output_dir_path"]
            logger.info(f"{building_id} を処理します")

            output_dir_path.mkdir(parents=True, exist_ok=True)
            byproduct_dir_path: Path | None = None
            if byproduct_root_dir_path is not None:
                byproduct_dir_path = byproduct_root_dir_path / building_id
                byproduct_dir_path.mkdir(parents=True, exist_ok=True)

            try:
                save_roofline_result(
                    corners=corners,
                    edges=edges,
                    rgb_file_path=sample["rgb_file_path"],
                    depth_file_path=sample["depth_file_path"],
                    input_rgb=sample["input_rgb"],
                    input_depth=sample["input_depth"],
                    output_dir_path=output_dir_path,
                    byproduct_dir_path=byproduct_dir_path,
                    backup=backup,
                )
            except Exception as e:
                logger.error(f"{building_id} の結果保存に失敗しました")
                if exit_on_error:
                    raise
                tb = traceback.format_exc()
                logger.exception(e)
                update_parameters(
                    output_dir_path / file_names.EXTRACT_ROOFLINE_OUTPUT,
                    {parameter_keys.ERROR: str(e), parameter_keys.TRACEBACK: tb},
                )
