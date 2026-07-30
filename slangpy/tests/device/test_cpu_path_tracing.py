# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
import slangpy as spy

from examples.cpu_path_tracing.cpu_path_tracing import RenderConfig, render


def test_cpu_path_tracing_cornell_box() -> None:
    probe_device = spy.Device(type=spy.DeviceType.cpu)
    try:
        if not probe_device.has_feature(spy.Feature.ray_query):
            pytest.skip("CPU RayQuery is not enabled in this slang-rhi build.")
    finally:
        probe_device.close()

    image = render(
        RenderConfig(
            width=48,
            height=48,
            spp=2,
            samples_per_pass=1,
            max_depth=3,
            output_path=None,
            enable_debug_layers=True,
            show_progress=False,
        )
    )

    assert image.shape == (48, 48, 4)
    assert image.dtype == np.float32
    assert np.all(np.isfinite(image))
    np.testing.assert_allclose(image[:, :, 3], 1.0)

    rgb = image[:, :, :3]
    visible_pixels = np.any(rgb > 0.0, axis=2)
    red_pixels = (
        (rgb[:, :, 0] > rgb[:, :, 1] * 1.35)
        & (rgb[:, :, 0] > rgb[:, :, 2] * 1.35)
        & (rgb[:, :, 0] > 0.05)
    )
    green_pixels = (
        (rgb[:, :, 1] > rgb[:, :, 0] * 1.35)
        & (rgb[:, :, 1] > rgb[:, :, 2] * 1.35)
        & (rgb[:, :, 1] > 0.05)
    )

    assert np.count_nonzero(visible_pixels) > 1000
    assert np.count_nonzero(red_pixels) > 100
    assert np.count_nonzero(green_pixels) > 100
