# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("transient", [False, True])
@pytest.mark.parametrize("increment", ["increment", "incrementQuad"])
@pytest.mark.parametrize(
    "policy", [spy.PipelineCompilationPolicy.immediate, spy.PipelineCompilationPolicy.deferred]
)
def test_persistent_tile_memory(
    device_type: spy.DeviceType,
    samples: int,
    transient: bool,
    increment: str,
    policy: spy.PipelineCompilationPolicy,
) -> None:
    device = helpers.get_device(type=device_type)
    if not device.has_feature(spy.Feature.tile_shading):
        pytest.skip("Tile shading not supported on this device")
    width, height = 37, 23
    column_offset = 5
    color = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        type=spy.TextureType.texture_2d_ms if samples > 1 else spy.TextureType.texture_2d,
        sample_count=samples,
        usage=spy.TextureUsage.render_target
        | (
            spy.TextureUsage.transient_attachment
            if transient and samples > 1
            else spy.TextureUsage.none
        ),
    )
    output = (
        device.create_texture(
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.render_target,
        )
        if samples > 1
        else color
    )
    depth = device.create_texture(
        format=spy.Format.r32_float,
        width=width,
        height=height,
        type=spy.TextureType.texture_2d_ms if samples > 1 else spy.TextureType.texture_2d,
        sample_count=samples,
        usage=spy.TextureUsage.render_target
        | (spy.TextureUsage.transient_attachment if transient else spy.TextureUsage.none),
    )
    tiles = [
        device.create_tile_pipeline(
            program=device.load_program("test_tile.slang", [entry]),
            color_formats=[spy.Format.rgba32_float, spy.Format.r32_float],
            sample_count=samples,
            compilation_policy=policy,
        )
        for entry in ["initialize", increment]
    ]
    assert tiles[0].thread_group_size == spy.uint3(16, 16, 1)
    graphics = device.create_render_pipeline(
        input_layout=None,
        program=device.load_program("test_tile.slang", ["vertexMain", "fragmentMain"]),
        targets=[
            {"format": spy.Format.rgba32_float},
            {"format": spy.Format.r32_float, "write_mask": spy.RenderTargetWriteMask.none},
        ],
        multisample={"sample_count": samples},
    )
    command = device.create_command_encoder()
    attachment = {"view": color.create_view({}), "clear_value": [0.25, 0, 0, 1]}
    if samples > 1:
        attachment["resolve_target"] = output.create_view({})
        attachment["store_op"] = spy.StoreOp.dont_care
    with command.begin_render_pass(
        {
            "color_attachments": [
                attachment,
                {
                    "view": depth.create_view({}),
                    "clear_value": [0.5, 0, 0, 0],
                    "store_op": spy.StoreOp.dont_care,
                },
            ],
            "tile_width": 16,
            "tile_height": 16,
            "tile_memory_sizes": [16, 16],
        }
    ) as encoder:
        cursor = spy.ShaderCursor(encoder.bind_pipeline(tiles[0]))
        cursor["tileInputs"]["columnOffset"] = column_offset
        encoder.dispatch_tile()
        encoder.bind_pipeline(tiles[1])
        encoder.dispatch_tile()
        encoder.bind_pipeline(graphics)
        encoder.set_render_state(
            {
                "viewports": [spy.Viewport.from_size(width, height)],
                "scissor_rects": [spy.ScissorRect.from_size(width, height)],
            }
        )
        encoder.draw({"vertex_count": 3})
    device.submit_command_buffer(command.finish())
    pixels = output.to_numpy()
    np.testing.assert_array_equal(pixels[:, :, 0], 0.75)
    np.testing.assert_array_equal(pixels[:, :, 1], 1)
    np.testing.assert_array_equal(
        pixels[:, :, 2], np.broadcast_to(np.arange(width) // 16 + column_offset, (height, width))
    )
    np.testing.assert_array_equal(
        pixels[:, :, 3], np.broadcast_to((np.arange(height) // 16 + 1)[:, None], (height, width))
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tile_rejects_compute_program(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(type=device_type)
    if not device.has_feature(spy.Feature.tile_shading):
        pytest.skip("Tile shading not supported on this device")
    module = device.load_module_from_source(
        "tile_wrong_stage", '[shader("compute")] [numthreads(1,1,1)] void main() {}'
    )
    program = device.link_program([module], [module.entry_point("main")])
    with pytest.raises(RuntimeError):
        device.create_tile_pipeline(
            program=program, color_formats=[spy.Format.rgba32_float, spy.Format.r32_float]
        )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("sizes", [[15], [16] * 32])
def test_tile_invalid_memory(device_type: spy.DeviceType, sizes: list[int]) -> None:
    device = helpers.get_device(type=device_type)
    if not device.has_feature(spy.Feature.tile_shading):
        pytest.skip("Tile shading not supported on this device")
    command = device.create_command_encoder()
    with pytest.raises(RuntimeError, match="Failed to begin render pass"):
        command.begin_render_pass({"tile_memory_sizes": sizes})
    command.finish()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("invalid", ["sampled", "initial_data", "load", "store", "resolve_target"])
def test_transient_attachment_validation(device_type: spy.DeviceType, invalid: str) -> None:
    if device_type != spy.DeviceType.metal:
        pytest.skip("Transient attachments are only backed by memoryless storage on Metal")
    device = helpers.get_device(type=device_type)
    usage = spy.TextureUsage.render_target | spy.TextureUsage.transient_attachment
    desc = dict(format=spy.Format.rgba32_float, width=16, height=16, usage=usage)
    if invalid in ["sampled", "initial_data"]:
        if invalid == "sampled":
            desc["usage"] |= spy.TextureUsage.shader_resource
        else:
            desc["data"] = np.zeros((16, 16, 4), dtype=np.float32)
        with pytest.raises(RuntimeError):
            device.create_texture(**desc)
        return
    texture = device.create_texture(**desc)
    attachment = dict(view=texture.create_view({}), store_op=spy.StoreOp.dont_care)
    if invalid == "load":
        attachment["load_op"] = spy.LoadOp.load
    elif invalid == "store":
        attachment["store_op"] = spy.StoreOp.store
    else:
        source = device.create_texture(
            format=spy.Format.rgba32_float,
            width=16,
            height=16,
            type=spy.TextureType.texture_2d_ms,
            sample_count=4,
            usage=spy.TextureUsage.render_target,
        )
        attachment["view"] = source.create_view({})
        attachment["resolve_target"] = texture.create_view({})
    command = device.create_command_encoder()
    with pytest.raises(RuntimeError, match="Failed to begin render pass"):
        command.begin_render_pass({"color_attachments": [attachment]})
    command.finish()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
