# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import slangpy as spy


EXAMPLE_DIR = Path(__file__).parent
DEFAULT_OUTPUT_PATH = EXAMPLE_DIR / "cornell_box.png"
DEFAULT_QUICK_OUTPUT_PATH = EXAMPLE_DIR / "cornell_box_quick.png"


@dataclass
class RenderConfig:
    """Configuration for a CPU Cornell Box render."""

    width: int = 512
    height: int = 512
    spp: int = 64
    samples_per_pass: int = 4
    max_depth: int = 5
    seed: int = 1
    output_path: Optional[Path] = DEFAULT_OUTPUT_PATH
    enable_debug_layers: bool = False
    show_progress: bool = True


@dataclass
class MeshRange:
    """Offsets and counts for one mesh in the shared geometry buffers."""

    vertex_offset: int
    vertex_count: int
    index_offset: int
    index_count: int


@dataclass
class SceneInstance:
    """A transformed BLAS instance and its diffuse/emissive material."""

    mesh_index: int
    transform: spy.float4x4
    material: tuple[float, float, float, float]


@dataclass
class SceneResources:
    """Resources that must remain alive while the path tracer runs."""

    tlas: spy.AccelerationStructure
    vertices: spy.Buffer
    indices: spy.Buffer
    instance_mesh_ranges: spy.Buffer
    instance_transforms: spy.Buffer
    instance_materials: spy.Buffer
    blases: list[spy.AccelerationStructure]


def _make_transform(
    translation: tuple[float, float, float],
    scaling: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> spy.float4x4:
    translation_matrix = spy.math.matrix_from_translation(spy.float3(translation))
    rotation_matrix = spy.math.matrix_from_rotation_xyz(spy.float3(rotation))
    scaling_matrix = spy.math.matrix_from_scaling(spy.float3(scaling))
    return spy.math.mul(spy.math.mul(translation_matrix, rotation_matrix), scaling_matrix)


def _create_geometry() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32], list[MeshRange]]:
    quad_vertices = np.array(
        [
            [-0.5, 0.0, -0.5],
            [0.5, 0.0, -0.5],
            [-0.5, 0.0, 0.5],
            [0.5, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    quad_indices = np.array([2, 1, 0, 1, 2, 3], dtype=np.uint32)

    cube_vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    cube_indices = np.array(
        [
            0,
            2,
            1,
            0,
            3,
            2,
            4,
            5,
            6,
            4,
            6,
            7,
            0,
            4,
            7,
            0,
            7,
            3,
            1,
            2,
            6,
            1,
            6,
            5,
            0,
            1,
            5,
            0,
            5,
            4,
            3,
            7,
            6,
            3,
            6,
            2,
        ],
        dtype=np.uint32,
    )

    vertices = np.concatenate([quad_vertices, cube_vertices], axis=0)
    indices = np.concatenate([quad_indices, cube_indices], axis=0)
    mesh_ranges = [
        MeshRange(
            vertex_offset=0,
            vertex_count=len(quad_vertices),
            index_offset=0,
            index_count=len(quad_indices),
        ),
        MeshRange(
            vertex_offset=len(quad_vertices),
            vertex_count=len(cube_vertices),
            index_offset=len(quad_indices),
            index_count=len(cube_indices),
        ),
    ]
    return vertices, indices, mesh_ranges


def _create_instances() -> list[SceneInstance]:
    white = (0.73, 0.73, 0.73, 0.0)
    red = (0.75, 0.06, 0.04, 0.0)
    green = (0.05, 0.62, 0.08, 0.0)
    light = (0.0, 0.0, 0.0, 1.0)

    return [
        SceneInstance(0, _make_transform((0.0, 0.0, 1.0), (2.0, 1.0, 2.0)), white),
        SceneInstance(
            0,
            _make_transform((0.0, 2.0, 1.0), (2.0, 1.0, 2.0), (math.pi, 0.0, 0.0)),
            white,
        ),
        SceneInstance(
            0,
            _make_transform(
                (0.0, 1.0, 2.0),
                (2.0, 1.0, 2.0),
                (-0.5 * math.pi, 0.0, 0.0),
            ),
            white,
        ),
        SceneInstance(
            0,
            _make_transform(
                (-1.0, 1.0, 1.0),
                (2.0, 1.0, 2.0),
                (0.0, 0.0, -0.5 * math.pi),
            ),
            red,
        ),
        SceneInstance(
            0,
            _make_transform(
                (1.0, 1.0, 1.0),
                (2.0, 1.0, 2.0),
                (0.0, 0.0, 0.5 * math.pi),
            ),
            green,
        ),
        SceneInstance(
            0,
            _make_transform(
                (0.0, 1.99, 1.0),
                (0.468, 1.0, 0.376),
                (math.pi, 0.0, 0.0),
            ),
            light,
        ),
        # The canonical Cornell Box short block is left/front: its measured footprint
        # is centered at (185.5, 169) and reaches 165 high in the 556x548.8x559.2 room.
        SceneInstance(
            1,
            _make_transform(
                (-0.333, 0.301, 0.604),
                (0.593, 0.601, 0.598),
                (0.0, math.radians(-16.5), 0.0),
            ),
            white,
        ),
        # The tall block is right/back, twice as high, with the opposite rotation.
        SceneInstance(
            1,
            _make_transform(
                (0.326, 0.601, 1.256),
                (0.595, 1.203, 0.595),
                (0.0, math.radians(17.1), 0.0),
            ),
            white,
        ),
    ]


def _build_blas(
    device: spy.Device,
    vertex_buffer: spy.Buffer,
    index_buffer: spy.Buffer,
    mesh_range: MeshRange,
    label: str,
) -> spy.AccelerationStructure:
    build_input = spy.AccelerationStructureBuildInputTriangles(
        {
            "vertex_buffers": [
                {
                    "buffer": vertex_buffer,
                    "offset": mesh_range.vertex_offset * 3 * np.dtype(np.float32).itemsize,
                }
            ],
            "vertex_format": spy.Format.rgb32_float,
            "vertex_count": mesh_range.vertex_count,
            "vertex_stride": 3 * np.dtype(np.float32).itemsize,
            "index_buffer": {
                "buffer": index_buffer,
                "offset": mesh_range.index_offset * np.dtype(np.uint32).itemsize,
            },
            "index_format": spy.IndexFormat.uint32,
            "index_count": mesh_range.index_count,
            "flags": spy.AccelerationStructureGeometryFlags.opaque,
        }
    )
    build_desc = spy.AccelerationStructureBuildDesc({"inputs": [build_input]})
    sizes = device.get_acceleration_structure_sizes(build_desc)
    scratch_buffer = device.create_buffer(
        size=sizes.scratch_size,
        usage=spy.BufferUsage.unordered_access,
        label=f"{label}_scratch",
    )
    blas = device.create_acceleration_structure(
        kind=spy.AccelerationStructureKind.bottom_level,
        size=sizes.acceleration_structure_size,
        label=label,
    )
    command_encoder = device.create_command_encoder()
    command_encoder.build_acceleration_structure(
        desc=build_desc,
        dst=blas,
        src=None,
        scratch_buffer=scratch_buffer,
    )
    device.submit_command_buffer(command_encoder.finish())
    device.wait()
    return blas


def _build_scene(device: spy.Device) -> SceneResources:
    vertices, indices, mesh_ranges = _create_geometry()
    instances = _create_instances()
    build_input_usage = (
        spy.BufferUsage.shader_resource | spy.BufferUsage.acceleration_structure_build_input
    )
    vertex_buffer = device.create_buffer(
        usage=build_input_usage,
        label="cornell_vertices",
        data=vertices,
    )
    index_buffer = device.create_buffer(
        usage=build_input_usage,
        label="cornell_indices",
        data=indices,
    )

    blases = [
        _build_blas(device, vertex_buffer, index_buffer, mesh_range, f"cornell_blas_{index}")
        for index, mesh_range in enumerate(mesh_ranges)
    ]

    instance_list = device.create_acceleration_structure_instance_list(len(instances))
    for index, instance in enumerate(instances):
        instance_list.write(
            index,
            {
                "transform": spy.float3x4(instance.transform),
                "instance_id": index,
                "instance_mask": 0xFF,
                "instance_contribution_to_hit_group_index": 0,
                "flags": spy.AccelerationStructureInstanceFlags.none,
                "acceleration_structure": blases[instance.mesh_index].handle,
            },
        )

    tlas_build_desc = spy.AccelerationStructureBuildDesc(
        {"inputs": [instance_list.build_input_instances()]}
    )
    tlas_sizes = device.get_acceleration_structure_sizes(tlas_build_desc)
    tlas_scratch = device.create_buffer(
        size=tlas_sizes.scratch_size,
        usage=spy.BufferUsage.unordered_access,
        label="cornell_tlas_scratch",
    )
    tlas = device.create_acceleration_structure(
        kind=spy.AccelerationStructureKind.top_level,
        size=tlas_sizes.acceleration_structure_size,
        label="cornell_tlas",
    )
    command_encoder = device.create_command_encoder()
    command_encoder.build_acceleration_structure(
        desc=tlas_build_desc,
        dst=tlas,
        src=None,
        scratch_buffer=tlas_scratch,
    )
    device.submit_command_buffer(command_encoder.finish())
    device.wait()

    instance_mesh_ranges = np.array(
        [
            (
                mesh_ranges[instance.mesh_index].vertex_offset,
                mesh_ranges[instance.mesh_index].index_offset,
            )
            for instance in instances
        ],
        dtype=np.uint32,
    )
    instance_transforms = np.stack(
        [instance.transform.to_numpy() for instance in instances]
    ).astype(np.float32)
    instance_materials = np.array(
        [instance.material for instance in instances],
        dtype=np.float32,
    )

    return SceneResources(
        tlas=tlas,
        vertices=vertex_buffer,
        indices=index_buffer,
        instance_mesh_ranges=device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="cornell_instance_mesh_ranges",
            data=instance_mesh_ranges,
        ),
        instance_transforms=device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="cornell_instance_transforms",
            data=instance_transforms,
        ),
        instance_materials=device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="cornell_instance_materials",
            data=instance_materials,
        ),
        blases=blases,
    )


def _validate_config(config: RenderConfig) -> None:
    if config.width <= 0 or config.height <= 0:
        raise ValueError("Render dimensions must be positive.")
    if config.spp <= 0:
        raise ValueError("Samples per pixel must be positive.")
    if config.samples_per_pass <= 0:
        raise ValueError("Samples per pass must be positive.")
    if config.max_depth <= 0:
        raise ValueError("Maximum path depth must be positive.")


def render(config: RenderConfig) -> npt.NDArray[np.float32]:
    """Render the procedural Cornell Box with the CPU RayQuery backend."""

    _validate_config(config)
    started = time.perf_counter()
    device = spy.Device(
        type=spy.DeviceType.cpu,
        enable_debug_layers=config.enable_debug_layers,
        compiler_options={"include_paths": [EXAMPLE_DIR]},
    )
    try:
        if not device.has_feature(spy.Feature.ray_query):
            raise RuntimeError(
                "CPU RayQuery is unavailable. Build slang-rhi with CPU RayQuery "
                "and TinyBVH support."
            )

        scene = _build_scene(device)
        trace_program = device.load_program("cpu_path_tracing.slang", ["trace_paths"])
        trace_kernel = device.create_compute_kernel(trace_program)
        tone_program = device.load_program("cpu_path_tracing.slang", ["tone_map"])
        tone_kernel = device.create_compute_kernel(tone_program)

        texture_usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
        accumulation_texture = device.create_texture(
            format=spy.Format.rgba32_float,
            width=config.width,
            height=config.height,
            usage=texture_usage,
            label="cornell_accumulation",
        )
        output_texture = device.create_texture(
            format=spy.Format.rgba32_float,
            width=config.width,
            height=config.height,
            usage=texture_usage,
            label="cornell_output",
        )

        scene_vars = {
            "tlas": scene.tlas,
            "vertices": scene.vertices,
            "indices": scene.indices,
            "instance_mesh_ranges": scene.instance_mesh_ranges,
            "instance_transforms": scene.instance_transforms,
            "instance_materials": scene.instance_materials,
        }
        sample_offset = 0
        pass_index = 0
        while sample_offset < config.spp:
            pass_started = time.perf_counter()
            pass_spp = min(config.samples_per_pass, config.spp - sample_offset)
            trace_kernel.dispatch(
                thread_count=[config.width, config.height, 1],
                vars={
                    "g_scene": scene_vars,
                    "g_accumulation": accumulation_texture,
                    "g_sample_offset": sample_offset,
                    "g_samples_per_pass": pass_spp,
                    "g_max_depth": config.max_depth,
                    "g_seed": config.seed,
                    "g_reset_accumulation": pass_index == 0,
                },
            )
            device.wait()
            sample_offset += pass_spp
            pass_index += 1
            if config.show_progress:
                pass_elapsed = time.perf_counter() - pass_started
                remaining_passes = math.ceil((config.spp - sample_offset) / config.samples_per_pass)
                print(
                    f"{sample_offset:4d}/{config.spp} spp "
                    f"({pass_elapsed:.2f}s, ETA {remaining_passes * pass_elapsed:.1f}s)"
                )

        tone_kernel.dispatch(
            thread_count=[config.width, config.height, 1],
            vars={
                "g_accumulation_input": accumulation_texture,
                "g_output": output_texture,
            },
        )
        device.wait()

        image = output_texture.to_numpy().astype(np.float32).reshape(config.height, config.width, 4)
        if not np.all(np.isfinite(image)):
            raise RuntimeError("Rendered image contains non-finite values.")

        if config.output_path is not None:
            output_path = Path(config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_texture.to_bitmap().convert(
                pixel_format=spy.Bitmap.PixelFormat.rgb,
                component_type=spy.Bitmap.ComponentType.uint8,
                srgb_gamma=True,
            ).write(output_path)
            if config.show_progress:
                print(f"Wrote {output_path.resolve()}")

        if config.show_progress:
            print(f"Total render time: {time.perf_counter() - started:.2f}s")
        return image
    finally:
        device.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Cornell Box with SlangPy's CPU RayQuery backend."
    )
    parser.add_argument("--width", type=int, default=512, help="Output width in pixels.")
    parser.add_argument("--height", type=int, default=512, help="Output height in pixels.")
    parser.add_argument("--spp", type=int, default=64, help="Samples per pixel.")
    parser.add_argument(
        "--samples-per-pass",
        type=int,
        default=4,
        help="Samples traced by each progressive pass.",
    )
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum path depth.")
    parser.add_argument("--seed", type=int, default=1, help="Deterministic RNG seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to cornell_box.png or cornell_box_quick.png.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable RHI debug layers.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Render a 256x256, 4 spp preview.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line CPU path tracer."""

    args = _parse_args()
    if args.quick:
        args.width = 256
        args.height = 256
        args.spp = 4
        args.samples_per_pass = 1
        args.max_depth = min(args.max_depth, 3)
    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_QUICK_OUTPUT_PATH if args.quick else DEFAULT_OUTPUT_PATH
    render(
        RenderConfig(
            width=args.width,
            height=args.height,
            spp=args.spp,
            samples_per_pass=args.samples_per_pass,
            max_depth=args.max_depth,
            seed=args.seed,
            output_path=output_path,
            enable_debug_layers=args.debug,
        )
    )


if __name__ == "__main__":
    main()
