from __future__ import annotations

import pytest

from industrial_defect_detection.config import Settings


def test_settings_rejects_non_stride_aligned_image_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFECT_IMAGE_SIZE", "641")

    with pytest.raises(ValueError, match="multiple of 32"):
        Settings.from_env().validate()


def test_settings_rejects_non_positive_pixel_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFECT_MAX_IMAGE_PIXELS", "0")

    with pytest.raises(ValueError, match="MAX_IMAGE_PIXELS"):
        Settings.from_env().validate()
