from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Annotated

import typer
from heat import HEAT
from loguru import logger
from torch.utils.data import DataLoader

from building_lod2_builder_heat.commands.extract_roofline.dataset import (
    RooflineDataset,
    RooflineSample,
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

_DATA_ROOT_HELP = (
    "データディレクトリのパス。各サブディレクトリの以下のファイルが使われます。\n"
    f"必須: {file_names.EXTRACT_ROOFLINE_PREPROCESS_RGB}。\n"
    "オプション: なし。"
)


@app.command()
def run(
    checkpoint_file_path: Annotated[
        Path, typer.Argument(help="学習済みモデルのパス。", exists=True)
    ],
    data_root_dir_path: Annotated[
        Path, typer.Argument(help=_DATA_ROOT_HELP, exists=True)
    ],
    output_root_dir_path: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="出力ディレクトリのパス。無指定の場合はデータディレクトリに出力する。",
        ),
    ] = None,
    byproduct_root_dir_path: Annotated[
        Path | None,
        typer.Option("--byproduct-dir", help="副産物を保存するディレクトリ。"),
    ] = None,
    prefer_gpu: Annotated[
        bool,
        typer.Option(
            "--prefer-gpu/--force-cpu",
            help="GPUが利用可能ならGPUを利用するかどうか。",
        ),
    ] = True,
    skip_exist: Annotated[
        bool,
        typer.Option(
            "--skip-exist/--overwrite",
            help="既に結果が存在する場合はスキップするかどうか。",
        ),
    ] = True,
    backup: Annotated[
        bool,
        typer.Option(
            "--backup",
            help="出力ファイルが既に存在する場合にバックアップを作成する。",
        ),
    ] = False,
    rich_error: Annotated[
        bool,
        typer.Option(
            "--rich-error/--normal-error",
            help="エラー時に変数の内容等まで出力するか。",
        ),
    ] = True,
    exit_on_error: Annotated[
        bool,
        typer.Option(
            "--exit-on-error",
            help="1つの処理対象に対するエラーで終了する。",
        ),
    ] = False,
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            help="ログレベルを指定します。",
            case_sensitive=False,
        ),
    ] = LogLevel.INFO,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="ログファイルの出力先パス。"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="一度に推論するサンプル数。", min=1),
    ] = 1,
    num_workers: Annotated[
        int,
        typer.Option("--num-workers", help="DataLoaderで使用するワーカー数。", min=0),
    ] = 0,
) -> None:
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
    if len(dataset) == 0:
        logger.info("処理対象がありません")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_samples,
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
            tb = traceback.format_exc()
            logger.exception(e)
            for sample in batch:
                _record_error(sample, e, tb)
            continue

        if len(batch_results) != len(batch):
            error = RuntimeError(
                "バッチ推論の入力数と結果数が一致しません: "
                f"{len(batch)} != {len(batch_results)}"
            )
            logger.error(str(error))
            if exit_on_error:
                raise error
            tb = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            for sample in batch:
                _record_error(sample, error, tb)
            continue

        # 各サンプルの結果を保存
        for sample, (corners, edges) in zip(batch, batch_results, strict=True):
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
                    input_rgb=sample["input_rgb"],
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
                _record_error(sample, e, tb)


def _collate_samples(batch: list[RooflineSample]) -> list[RooflineSample]:
    """
    DataLoader のバッチをそのままリストとして返す。

    :param batch: サンプルのリスト。
    :returns: 入力と同じサンプルのリスト。
    """
    return batch


def _record_error(
    sample: RooflineSample, error: Exception, traceback_text: str
) -> None:
    """
    対象ごとの処理エラーを出力パラメータに記録する。

    :param sample: 処理対象のサンプル。
    :param error: 記録する例外。
    :param traceback_text: 記録するトレースバック文字列。
    """
    output_dir_path = sample["output_dir_path"]
    update_parameters(
        output_dir_path / file_names.EXTRACT_ROOFLINE_OUTPUT,
        {parameter_keys.ERROR: str(error), parameter_keys.TRACEBACK: traceback_text},
        overwrite=True,
    )
