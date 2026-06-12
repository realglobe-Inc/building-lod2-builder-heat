from __future__ import annotations

import logging

import pytest

from building_lod2_builder_heat.common.logging import LogLevel, setup_logger


def test_standard_logging_error_is_visible_at_error_level(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    標準 logging の ERROR を WARNING 以下に落とさない。
    """
    setup_logger(LogLevel.ERROR)

    logging.error("visible-error")

    captured = capsys.readouterr()
    assert "visible-error" in captured.err
