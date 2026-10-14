# Forward+ lighting with Metal tile shaders

This sample demonstrates single-pass Forward+ tiled lighting in SlangPy using Metal tile shaders.
It builds a procedural courtyard of pillars, stepped plinths, and glossy spheres lit by 512
animated point lights — with no external model or texture assets required:

1. Draw the scene into the depth attachment and a depth color attachment, which tile shaders
   read through `MetalImageBlock<Pixel>`.
2. Dispatch the `reduceDepth` tile shader to compute each tile's `[minZ, maxZ]` depth bounds
   across all pixels and MSAA samples.
3. Dispatch the `cullLights` tile shader to test each point light against the tile's frustum and
   depth bounds, recording visible light indices in tile memory (`SV_MetalThreadgroup0`).
4. Draw the scene again, shading each fragment using only its tile's visible light list from tile
   memory, followed by billboard light markers.

The per-tile light lists live in on-chip tile memory (`TileLightsPtr : SV_MetalThreadgroup0`),
shared directly between the tile shaders and `lightingMain` without touching device memory.
The depth attachments and multisample color attachment use `TextureUsage.transient_attachment`,
so they stay in tile memory as well.

## Requirements

- macOS with an Apple GPU that supports `Feature.tile_shading`.
- `pillow`, for writing the output PNG in headless mode.

## Running

```sh
python forward_plus.py --interactive
python forward_plus.py --verify
python forward_plus.py --width 641 --height 433 --samples 1 --time 2.5 --verify
python forward_plus.py --heatmap
```

In interactive mode, `Space` pauses the animation, `H` toggles the per-tile light-count heatmap,
`C` toggles tile light culling, and `Escape` closes the window.

In headless mode, the sample renders one frame and writes a tone-mapped PNG plus a JSON report of
per-tile light statistics to `--output` (`output/forward-plus.png` by default). `--verify` renders
the scene both with tile culling enabled and with brute-force all-lights evaluation, verifying
that the floating-point outputs match within `2e-4`.
