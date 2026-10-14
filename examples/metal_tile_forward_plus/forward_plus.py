# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Forward+ tiled lighting with Metal tile shaders in a single SlangPy render pass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import slangpy as spy

HERE = Path(__file__).resolve().parent
LIGHT_COUNT = 512


def rotation(angle: float, axis: str) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], np.float32)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], np.float32)


def make_view(seconds: float) -> np.ndarray:
    """Orbiting camera looking toward the courtyard center (left-handed view space, +Z forward)."""
    translation = np.eye(4, dtype=np.float32)
    translation[:3, 3] = [0.0, -3.2, 24.0]
    return translation @ rotation(-0.42, "x") @ rotation(seconds * 0.25 + 0.35, "y")


def add_box(
    parts: list[np.ndarray],
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> None:
    cx, cy, cz = center
    hx, hy, hz = half_extents
    faces = [
        ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ((0, 0, -1), [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)]),
        ((1, 0, 0), [(hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)]),
        ((-1, 0, 0), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
        ((0, 1, 0), [(-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz)]),
        ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
    ]
    tris = np.zeros((12, 3, 3, 4), dtype=np.float32)
    for f_idx, (normal, corners) in enumerate(faces):
        for t_idx, tri in enumerate([(0, 1, 2), (0, 2, 3)]):
            row = f_idx * 2 + t_idx
            for v_idx, corner_idx in enumerate(tri):
                px, py, pz = corners[corner_idx]
                tris[row, v_idx, 0] = [cx + px, cy + py, cz + pz, 1.0]
                tris[row, v_idx, 1] = [normal[0], normal[1], normal[2], 0.0]
                tris[row, v_idx, 2] = color
    parts.append(tris.reshape(-1, 12))


def add_sphere(
    parts: list[np.ndarray],
    center: tuple[float, float, float],
    radius: float,
    color: tuple[float, float, float, float],
    lat_steps: int = 20,
    lon_steps: int = 32,
) -> None:
    cx, cy, cz = center
    thetas = np.linspace(0.0, np.pi, lat_steps + 1, dtype=np.float32)
    phis = np.linspace(0.0, 2.0 * np.pi, lon_steps + 1, dtype=np.float32)
    grid_theta, grid_phi = np.meshgrid(thetas, phis, indexing="ij")
    nx = np.sin(grid_theta) * np.cos(grid_phi)
    ny = np.cos(grid_theta)
    nz = np.sin(grid_theta) * np.sin(grid_phi)
    normals = np.stack([nx, ny, nz], axis=-1)
    positions = normals * radius + np.array([cx, cy, cz], dtype=np.float32)

    p00, p10 = positions[:-1, :-1], positions[1:, :-1]
    p11, p01 = positions[1:, 1:], positions[:-1, 1:]
    n00, n10 = normals[:-1, :-1], normals[1:, :-1]
    n11, n01 = normals[1:, 1:], normals[:-1, 1:]

    tri_p = np.concatenate(
        [np.stack([p00, p10, p11], axis=2), np.stack([p00, p11, p01], axis=2)], axis=0
    ).reshape(-1, 3, 3)
    tri_n = np.concatenate(
        [np.stack([n00, n10, n11], axis=2), np.stack([n00, n11, n01], axis=2)], axis=0
    ).reshape(-1, 3, 3)

    verts = np.zeros((tri_p.shape[0], 3, 3, 4), dtype=np.float32)
    verts[:, :, 0, :3] = tri_p
    verts[:, :, 0, 3] = 1.0
    verts[:, :, 1, :3] = tri_n
    verts[:, :, 2] = color
    parts.append(verts.reshape(-1, 12))


def build_scene() -> np.ndarray:
    """Build a self-contained courtyard of pillars, stepped plinths, and spheres."""
    parts: list[np.ndarray] = []

    stone = (0.72, 0.74, 0.78, 0.25)
    warm_stone = (0.82, 0.78, 0.72, 0.35)
    pillar_color = (0.78, 0.80, 0.84, 0.40)
    gold_gloss = (0.92, 0.78, 0.52, 0.90)
    jade_gloss = (0.45, 0.80, 0.74, 0.85)
    coral_gloss = (0.88, 0.55, 0.52, 0.85)

    # Stepped floor and central dais.
    add_box(parts, (0.0, -0.4, 0.0), (16.0, 0.4, 16.0), stone)
    add_box(parts, (0.0, 0.2, 0.0), (10.5, 0.2, 10.5), warm_stone)
    add_box(parts, (0.0, 0.6, 0.0), (5.2, 0.2, 5.2), stone)
    add_box(parts, (0.0, 1.3, 0.0), (2.2, 0.5, 2.2), warm_stone)

    # Central hero sphere and four surrounding pedestals with glossy spheres.
    add_sphere(parts, (0.0, 3.6, 0.0), 1.8, gold_gloss, lat_steps=28, lon_steps=44)
    for i in range(4):
        angle = i * (np.pi / 2.0) + np.pi / 4.0
        px, pz = 5.4 * np.cos(angle), 5.4 * np.sin(angle)
        add_box(parts, (px, 1.35, pz), (0.85, 0.95, 0.85), warm_stone)
        sphere_color = jade_gloss if i % 2 == 0 else coral_gloss
        add_sphere(parts, (px, 3.25, pz), 0.95, sphere_color, lat_steps=20, lon_steps=32)

    # Outer colonnade of 12 tall pillars + lintel ring (creates strong depth discontinuities).
    ring_radius = 9.2
    for i in range(12):
        angle = i * (2.0 * np.pi / 12.0)
        px, pz = ring_radius * np.cos(angle), ring_radius * np.sin(angle)
        add_box(parts, (px, 3.4, pz), (0.55, 3.0, 0.55), pillar_color)
        add_box(parts, (px, 0.6, pz), (0.80, 0.25, 0.80), warm_stone)
        add_box(parts, (px, 6.55, pz), (0.80, 0.25, 0.80), warm_stone)

    # Outer low perimeter walls and corner obelisks.
    for sx in (-1.0, 1.0):
        add_box(parts, (sx * 13.5, 1.1, 0.0), (0.45, 1.1, 13.0), stone)
        add_box(parts, (0.0, 1.1, sx * 13.5), (13.0, 1.1, 0.45), stone)
        for sz in (-1.0, 1.0):
            add_box(parts, (sx * 13.5, 2.6, sz * 13.5), (0.85, 2.6, 0.85), pillar_color)
            add_sphere(parts, (sx * 13.5, 5.8, sz * 13.5), 0.65, gold_gloss, 16, 24)

    return np.concatenate(parts, axis=0)


def make_lights(view: np.ndarray, seconds: float) -> np.ndarray:
    """Generate 512 animated point lights arranged in four concentric tiers."""
    rng = np.random.default_rng(0x51A6)
    lights = np.zeros((LIGHT_COUNT, 3, 4), dtype=np.float32)

    palette = np.array(
        [
            [1.00, 0.38, 0.20],
            [0.20, 0.78, 1.00],
            [1.00, 0.82, 0.24],
            [0.45, 0.95, 0.55],
            [0.92, 0.35, 0.88],
            [0.40, 0.52, 1.00],
        ],
        dtype=np.float32,
    )

    for i in range(LIGHT_COUNT):
        if i < 128:
            # Inner swirl around the central sphere and pedestals.
            dist = rng.uniform(2.8, 6.8)
            height = rng.uniform(1.0, 4.8)
            speed = rng.uniform(0.35, 0.75)
            radius = rng.uniform(2.0, 2.8)
        elif i < 320:
            # Mid-ring weaving through the 12 pillars.
            dist = rng.uniform(7.2, 10.8)
            height = rng.uniform(0.8, 6.2)
            speed = rng.uniform(-0.55, -0.20)
            radius = rng.uniform(2.2, 3.0)
        elif i < 448:
            # Low ground skimmers across the courtyard floor and perimeter wall.
            dist = rng.uniform(10.5, 14.2)
            height = rng.uniform(0.45, 2.2)
            speed = rng.uniform(0.18, 0.45)
            radius = rng.uniform(2.2, 3.0)
        else:
            # High canopy crown above the central dome.
            dist = rng.uniform(1.5, 5.0)
            height = rng.uniform(5.2, 7.6)
            speed = rng.uniform(0.4, 0.9)
            radius = rng.uniform(2.4, 3.2)

        base_angle = rng.uniform(0.0, 2.0 * np.pi)
        bob_phase = rng.uniform(0.0, 2.0 * np.pi)
        angle = base_angle + seconds * speed
        y = height + 0.35 * np.sin(seconds * 1.8 + bob_phase)

        lights[i, 0] = [dist * np.sin(angle), y, dist * np.cos(angle), radius]
        tint = palette[i % len(palette)] * rng.uniform(0.65, 1.15)
        lights[i, 1, :3] = tint

    homogeneous = lights[:, 0].copy()
    homogeneous[:, 3] = 1.0
    lights[:, 2] = homogeneous @ view.T
    lights[:, 2, 3] = lights[:, 0, 3]
    return lights


class Renderer:
    def __init__(self, width: int, height: int, samples: int = 4, seconds: float = 0.0) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        self.width, self.height, self.samples = width, height, samples
        self.device = spy.Device(
            type=spy.DeviceType.metal,
            enable_debug_layers=True,
            compiler_options={"include_paths": [HERE]},
        )
        if not self.device.has_feature(spy.Feature.tile_shading):
            raise RuntimeError("This sample requires a Metal GPU with native tile shading")

        vertices = build_scene()
        self.vertex_count = len(vertices)
        self.vertex_buffer = self.device.create_buffer(
            data=vertices, usage=spy.BufferUsage.shader_resource
        )
        self.sampler = self.device.create_sampler(
            {
                "min_filter": spy.TextureFilteringMode.linear,
                "mag_filter": spy.TextureFilteringMode.linear,
            }
        )

        view = make_view(seconds)
        sy = 1.0 / np.tan(np.radians(55.0) / 2.0)
        near, far = 0.5, 80.0
        a = far / (far - near)
        b = -(far * near) / (far - near)
        projection = np.array(
            [[sy * height / width, 0, 0, 0], [0, sy, 0, 0], [0, 0, a, b], [0, 0, 1, 0]], np.float32
        )
        self.uniforms = {
            "view": spy.float4x4(view),
            "projection": spy.float4x4(projection),
            "cameraPosition": spy.float3(np.linalg.inv(view)[:3, 3]),
            "lightCount": LIGHT_COUNT,
            # The render pass supplies `tilePass.imageBlock`, so only the constants are set.
            "tilePass": {
                "imageSize": spy.uint2(width, height),
                "depthUnproject": spy.float2(a, b),
                "screenToView": spy.float3(2 / sy / height, -width / height / sy, -1 / sy),
            },
        }
        self.lights = make_lights(view, seconds)
        self.counts = self.device.create_buffer(
            size=((width + 15) // 16) * ((height + 15) // 16) * 4,
            usage=spy.BufferUsage.unordered_access,
        )
        self.color = self.texture(spy.Format.rgba32_float, samples, transient=samples > 1)
        self.depth_color = self.texture(spy.Format.r32_float, samples, transient=True)
        self.depth = self.texture(spy.Format.d32_float, samples, depth=True, transient=True)
        self.output = self.texture(spy.Format.rgba32_float, 1) if samples > 1 else self.color

        # Every pipeline in the pass must agree on the attachment formats and sample count,
        # including the tile pipelines, which access the attachments through the imageblock.
        self.graphics: dict[str, spy.RenderPipeline] = {}
        for vertex, fragment in [
            ("depthVertex", "depthMain"),
            ("vertexMain", "lightingMain"),
            ("unlitVertex", "unlitMain"),
        ]:
            self.graphics[fragment] = self.device.create_render_pipeline(
                input_layout=None,
                program=self.device.load_program("forward_plus.slang", [vertex, fragment]),
                targets=[
                    {"format": spy.Format.rgba32_float},
                    {
                        "format": spy.Format.r32_float,
                        "write_mask": (
                            spy.RenderTargetWriteMask.all
                            if fragment == "depthMain"
                            else spy.RenderTargetWriteMask.none
                        ),
                    },
                ],
                depth_stencil={
                    "format": spy.Format.d32_float,
                    "depth_test_enable": True,
                    "depth_write_enable": fragment == "depthMain",
                    "depth_func": spy.ComparisonFunc.less_equal,
                },
                rasterizer={"cull_mode": spy.CullMode.none},
                multisample={"sample_count": samples},
            )
        self.tiles: dict[str, spy.TilePipeline] = {}
        for entry in ["reduceDepth", "cullLights"]:
            self.tiles[entry] = self.device.create_tile_pipeline(
                program=self.device.load_program("forward_plus.slang", [entry]),
                color_formats=[spy.Format.rgba32_float, spy.Format.r32_float],
                sample_count=samples,
            )

    def update(self, seconds: float) -> None:
        view = make_view(seconds)
        self.uniforms["view"] = spy.float4x4(view)
        self.uniforms["cameraPosition"] = spy.float3(np.linalg.inv(view)[:3, 3])
        self.lights = make_lights(view, seconds)

    def texture(
        self, format: spy.Format, samples: int, depth: bool = False, transient: bool = False
    ) -> spy.Texture:
        usage = spy.TextureUsage.depth_stencil if depth else spy.TextureUsage.render_target
        if transient:
            usage |= spy.TextureUsage.transient_attachment
        else:
            usage |= spy.TextureUsage.shader_resource
        return self.device.create_texture(
            format=format,
            width=self.width,
            height=self.height,
            type=spy.TextureType.texture_2d_ms if samples > 1 else spy.TextureType.texture_2d,
            sample_count=samples,
            usage=usage,
        )

    def bind(
        self, encoder: spy.RenderPassEncoder, pipeline: spy.RenderPipeline | spy.TilePipeline
    ) -> spy.ShaderCursor:
        """Bind a pipeline and set the module-scope parameters that its entry points use."""
        cursor = spy.ShaderCursor(encoder.bind_pipeline(pipeline))
        for name, value in {
            **self.uniforms,
            "vertices": self.vertex_buffer,
            "tileCounts": self.counts,
        }.items():
            if cursor.has_field(name):
                cursor[name] = value
        if cursor.has_field("lights"):
            cursor["lights"].dereference().set_data(self.lights)
        return cursor

    def draw_scene(self, encoder: spy.RenderPassEncoder, fragment: str) -> None:
        self.bind(encoder, self.graphics[fragment])
        encoder.draw({"vertex_count": self.vertex_count})

    def dispatch_tile(self, encoder: spy.RenderPassEncoder, entry: str) -> None:
        self.bind(encoder, self.tiles[entry])
        encoder.dispatch_tile()

    def render(
        self,
        culling: bool = True,
        heatmap: bool = False,
        unlit: bool = True,
        readback: bool = True,
    ) -> np.ndarray | None:
        self.uniforms["useCulling"] = int(culling)
        self.uniforms["showHeatmap"] = int(heatmap)
        command = self.device.create_command_encoder()
        color = {"view": self.color.create_view({}), "clear_value": [0.015, 0.02, 0.035, 1.0]}
        if self.samples > 1:
            color["resolve_target"] = self.output.create_view({})
            color["store_op"] = spy.StoreOp.dont_care
        with command.begin_render_pass(
            {
                "color_attachments": [
                    color,
                    {
                        "view": self.depth_color.create_view({}),
                        "clear_value": [1.0, 0.0, 0.0, 0.0],
                        "store_op": spy.StoreOp.dont_care,
                    },
                ],
                "depth_stencil_attachment": {
                    "view": self.depth.create_view({}),
                    "depth_store_op": spy.StoreOp.dont_care,
                },
                "tile_width": 16,
                "tile_height": 16,
                # `TileLights` takes 4 * (3 + 64) bytes, rounded up to the 16-byte granularity
                # of Metal tile memory.
                "tile_memory_sizes": [272],
            }
        ) as encoder:
            encoder.set_render_state(
                {
                    "viewports": [spy.Viewport.from_size(self.width, self.height)],
                    "scissor_rects": [spy.ScissorRect.from_size(self.width, self.height)],
                }
            )
            # Draws and tile dispatches interleave within the pass. Each tile dispatch runs
            # after all earlier draws have finished with the tile, and its tile memory writes
            # are visible to the fragment shaders of later draws.
            self.draw_scene(encoder, "depthMain")
            self.dispatch_tile(encoder, "reduceDepth")
            self.dispatch_tile(encoder, "cullLights")
            self.draw_scene(encoder, "lightingMain")
            if unlit:
                self.bind(encoder, self.graphics["unlitMain"])
                encoder.draw({"vertex_count": 6, "instance_count": LIGHT_COUNT})
        self.device.submit_command_buffer(command.finish())
        return self.output.to_numpy() if readback else None


def save_image(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    color = np.maximum(pixels[:, :, :3], 0)
    color = color / (1 + color)
    color = np.where(color <= 0.0031308, 12.92 * color, 1.055 * color ** (1 / 2.4) - 0.055)
    Image.fromarray(np.uint8(np.clip(color * 255, 0, 255))).save(path)


def run_interactive(renderer: Renderer, seconds: float, frame_limit: int) -> None:
    window = spy.Window(
        width=renderer.width,
        height=renderer.height,
        title="SlangPy Metal Forward+ | Space: pause  H: heatmap  C: culling  Esc: close",
        resizable=False,
    )
    surface = renderer.device.create_surface(window)
    surface.configure(
        width=renderer.width, height=renderer.height, format=spy.Format.bgra8_unorm, vsync=True
    )
    pipeline = renderer.device.create_render_pipeline(
        input_layout=None,
        program=renderer.device.load_program("present.slang", ["vertexMain", "fragmentMain"]),
        targets=[{"format": spy.Format.bgra8_unorm}],
    )
    state = {"paused": False, "heatmap": False, "culling": True}

    def keyboard(event: spy.KeyboardEvent) -> None:
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                window.close()
            for key, field in [
                (spy.KeyCode.space, "paused"),
                (spy.KeyCode.h, "heatmap"),
                (spy.KeyCode.c, "culling"),
            ]:
                if event.key == key:
                    state[field] = not state[field]

    window.on_keyboard_event = keyboard
    previous = time.perf_counter()
    frame = 0
    while not window.should_close() and (frame_limit <= 0 or frame < frame_limit):
        window.process_events()
        now = time.perf_counter()
        if not state["paused"]:
            seconds += now - previous
        previous = now
        target = surface.acquire_next_image()
        if target is None:
            continue
        renderer.update(seconds)
        renderer.render(culling=state["culling"], heatmap=state["heatmap"], readback=False)
        command = renderer.device.create_command_encoder()
        with command.begin_render_pass(
            {"color_attachments": [{"view": target.create_view({})}]}
        ) as encoder:
            cursor = spy.ShaderCursor(encoder.bind_pipeline(pipeline))
            cursor["source"] = renderer.output
            cursor["sourceSampler"] = renderer.sampler
            cursor["outputSize"] = spy.uint2(target.width, target.height)
            encoder.set_render_state(
                {
                    "viewports": [spy.Viewport.from_size(target.width, target.height)],
                    "scissor_rects": [spy.ScissorRect.from_size(target.width, target.height)],
                }
            )
            encoder.draw({"vertex_count": 3})
        renderer.device.submit_command_buffer(command.finish())
        del target
        surface.present()
        frame += 1
    renderer.device.wait()
    surface.unconfigure()
    window.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--samples", type=int, choices=[1, 4], default=4)
    parser.add_argument("--time", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=HERE / "output/forward-plus.png")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--frames", type=int, default=0, help="Limit interactive frames for smoke tests"
    )
    args = parser.parse_args()
    start = time.perf_counter()
    renderer = Renderer(args.width, args.height, args.samples, args.time)
    if args.interactive:
        run_interactive(renderer, args.time, args.frames)
        return
    pixels = renderer.render(heatmap=args.heatmap)
    assert pixels is not None
    if not np.isfinite(pixels).all() or np.std(pixels[:, :, :3]) < 0.01:
        raise RuntimeError("Rendering produced invalid or empty output")
    save_image(args.output, pixels)
    counts = renderer.counts.to_numpy().view(np.uint32)
    report = {
        "width": args.width,
        "height": args.height,
        "samples": args.samples,
        "lights": LIGHT_COUNT,
        "max_lights_per_tile": int(counts.max()),
        "mean_lights_per_tile": float(counts.mean()),
        "overflow_tiles": int((counts > 64).sum()),
        "output": str(args.output.resolve()),
    }
    if args.verify:
        actual = renderer.render(unlit=False)
        expected = renderer.render(culling=False, unlit=False)
        assert actual is not None and expected is not None
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError("Nonfinite pixels in lighting comparison")
        difference = np.abs(actual - expected)
        report["max_absolute_error"] = float(difference.max())
        report["mean_absolute_error"] = float(difference.mean())
        if not np.allclose(actual, expected, rtol=2e-4, atol=2e-4):
            save_image(args.output.with_stem(args.output.stem + "-reference"), expected)
            raise AssertionError(f"Tile lighting differs from all-lights reference: {report}")
    report["elapsed_seconds"] = time.perf_counter() - start
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
