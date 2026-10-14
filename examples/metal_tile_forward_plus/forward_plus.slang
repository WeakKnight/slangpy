// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Forward+ tiled lighting in a single render pass:
//   1. `depthMain` writes the depth buffer and a depth color attachment in the imageblock.
//   2. `reduceDepth` (tile shader) reduces each tile's MSAA depth samples into [minZ, maxZ].
//   3. `cullLights` (tile shader) intersects point lights with the tile frustum in tile memory.
//   4. `lightingMain` shades fragments from the per-tile light list in `SV_MetalThreadgroup0`.

static const uint tileSize = 16;
static const uint maxLightsPerTile = 64;

// The light list of a tile, stored in tile memory slot 0 (`SV_MetalThreadgroup0`).
// Depth bounds are stored as float bits so integer atomics can reduce them; normalized device
// depths in [0, 1] are nonnegative, so unsigned integer order matches float order.
struct TileLights
{
    uint count; // Total lights overlapping the tile (may exceed `maxLightsPerTile`).
    uint minDepthBits;
    uint maxDepthBits;
    uint indices[maxLightsPerTile];
};

typealias TileLightsPtr = Ptr<TileLights, Access.ReadWrite, AddressSpace.GroupShared>;

struct Vertex
{
    float4 position;
    float4 normal;
    float4 color; // rgb = albedo, a = specular intensity
};

struct Light
{
    float4 position;     // xyz = world position, w = radius
    float4 color;        // rgb = radiance
    float4 viewPosition; // xyz = view-space position, w = radius
};

struct LightTable
{
    Light values[512];
};

struct Pixel
{
    float4 lighting : SV_Target0;
    float depth : SV_Target1;
};

struct Varyings
{
    float4 position : SV_Position;
    float3 worldPosition;
    float3 normal;
    float4 color;
};

StructuredBuffer<Vertex> vertices;
ConstantBuffer<LightTable> lights;
RWStructuredBuffer<uint> tileCounts;

uniform float4x4 view;
uniform float4x4 projection;
uniform float3 cameraPosition;
uniform uint lightCount;
uniform uint useCulling;
uniform uint showHeatmap;

// Parameters of the tile shaders. The render pass supplies `imageBlock` implicitly, so it sits
// in the uniform struct next to ordinary constants.
struct TilePass
{
    MetalImageBlock<Pixel> imageBlock;
    uint2 imageSize;
    float2 depthUnproject;
    float3 screenToView;
};

uniform TilePass tilePass;

[shader("vertex")]
float4 depthVertex(uint vertexID : SV_VertexID)
    : SV_Position
{
    return mul(projection, mul(view, float4(vertices[vertexID].position.xyz, 1.0f)));
}

[shader("fragment")]
Pixel depthMain(float4 position : SV_Position)
{
    Pixel p;
    p.lighting = float4(0.0f, 0.0f, 0.0f, 1.0f);
    p.depth = position.z;
    return p;
}

[shader("tile")]
[numthreads(16, 16, 1)]
void reduceDepth(
    TileLightsPtr tile : SV_MetalThreadgroup0,
    uint2 local : SV_GroupThreadID,
    uint3 group : SV_GroupID,
    uint index : SV_GroupIndex)
{
    if (index == 0)
    {
        tile->count = 0;
        tile->minDepthBits = asuint(1.0f);
        tile->maxDepthBits = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    float lo = 1.0f;
    float hi = 0.0f;
    bool inside = all(group.xy * tileSize + local < tilePass.imageSize) &&
                  local.x < tilePass.imageBlock.getWidth() &&
                  local.y < tilePass.imageBlock.getHeight();
    if (inside)
    {
        for (uint s = 0; s < tilePass.imageBlock.getSampleCount(); ++s)
        {
            float z = tilePass.imageBlock.readSample(local, s).depth;
            lo = min(lo, z);
            hi = max(hi, z);
        }
    }

    // Reduce within each 2x2 quad first, then update tile memory once per quad.
    lo = min(lo, QuadReadAcrossX(lo));
    lo = min(lo, QuadReadAcrossY(lo));
    hi = max(hi, QuadReadAcrossX(hi));
    hi = max(hi, QuadReadAcrossY(hi));
    if (index == QuadReadLaneAt(index, 0))
    {
        InterlockedMin(tile->minDepthBits, asuint(lo));
        InterlockedMax(tile->maxDepthBits, asuint(hi));
    }
}

[shader("tile")]
[numthreads(16, 16, 1)]
void cullLights(
    TileLightsPtr tile : SV_MetalThreadgroup0,
    uint3 group : SV_GroupID,
    uint index : SV_GroupIndex)
{
    float2 depthUnproject = tilePass.depthUnproject;
    float3 screenToView = tilePass.screenToView;
    float nearZ = depthUnproject.y / (asfloat(tile->minDepthBits) - depthUnproject.x);
    float farZ = depthUnproject.y / (asfloat(tile->maxDepthBits) - depthUnproject.x);
    float2 lo = float2(group.xy * tileSize) * float2(screenToView.x, -screenToView.x) +
                float2(screenToView.y, -screenToView.z);
    float2 hi = float2((group.xy + 1) * tileSize) * float2(screenToView.x, -screenToView.x) +
                float2(screenToView.y, -screenToView.z);
    float3 planes[4] = {
        normalize(float3(1.0f, 0.0f, -hi.x)),
        normalize(float3(0.0f, 1.0f, -lo.y)),
        normalize(float3(-1.0f, 0.0f, lo.x)),
        normalize(float3(0.0f, -1.0f, hi.y)),
    };

    for (uint id = index; id < lightCount; id += tileSize * tileSize)
    {
        float4 sphere = lights.values[id].viewPosition;
        float3 p = sphere.xyz;
        float radius = sphere.w;
        bool visible = p.z + radius >= nearZ && p.z - radius <= farZ;
        for (uint i = 0; i < 4; ++i)
            visible = visible && dot(planes[i], p) <= radius;
        if (visible)
        {
            uint slot;
            InterlockedAdd(tile->count, 1, slot);
            if (slot < maxLightsPerTile)
                tile->indices[slot] = id;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (index == 0)
    {
        uint tilesPerRow = (tilePass.imageSize.x + tileSize - 1) / tileSize;
        tileCounts[group.y * tilesPerRow + group.x] = tile->count;
    }
}

[shader("vertex")]
Varyings vertexMain(uint vertexID : SV_VertexID)
{
    Vertex v = vertices[vertexID];
    Varyings o;
    o.position = mul(projection, mul(view, float4(v.position.xyz, 1.0f)));
    o.worldPosition = v.position.xyz;
    o.normal = v.normal.xyz;
    o.color = v.color;
    return o;
}

float3 heatmapColor(float t)
{
    float3 c0 = float3(0.08f, 0.12f, 0.35f);
    float3 c1 = float3(0.10f, 0.75f, 0.65f);
    float3 c2 = float3(0.98f, 0.82f, 0.20f);
    float3 c3 = float3(0.95f, 0.20f, 0.15f);
    if (t < 0.33f)
        return lerp(c0, c1, t / 0.33f);
    if (t < 0.66f)
        return lerp(c1, c2, (t - 0.33f) / 0.33f);
    return lerp(c2, c3, saturate((t - 0.66f) / 0.34f));
}

[shader("fragment")]
float4 lightingMain(Varyings v, TileLightsPtr tile : SV_MetalThreadgroup0)
    : SV_Target0
{
    uint count = tile->count;
    float3 n = normalize(v.normal);
    float3 V = normalize(cameraPosition - v.worldPosition);

    // Subtle procedural floor grid to give spatial scale without external textures.
    float3 albedo = v.color.rgb;
    if (abs(n.y) > 0.9f)
    {
        float2 grid = abs(frac(v.worldPosition.xz * 0.5f - 0.5f) - 0.5f);
        float lineMask = smoothstep(0.0f, 0.04f, min(grid.x, grid.y));
        albedo *= lerp(0.72f, 1.0f, lineMask);
    }

    if (showHeatmap != 0)
    {
        float t = min(float(count) / float(maxLightsPerTile), 1.0f);
        float3 heat = heatmapColor(t);
        float3 shaded = albedo * (0.25f + 0.75f * max(dot(n, normalize(float3(-0.4f, 1.0f, -0.3f))), 0.0f));
        return float4(lerp(shaded * 0.25f, heat, 0.82f), 1.0f);
    }

    float specularStrength = v.color.a;
    float3 sunDir = normalize(float3(-0.45f, 0.85f, -0.35f));
    float3 color = albedo * (0.04f + 0.12f * max(dot(n, sunDir), 0.0f));

    // A tile whose list overflowed evaluates every light, so overflow costs time but not correctness.
    bool culled = useCulling != 0 && count <= maxLightsPerTile;
    uint iterations = culled ? count : lightCount;
    for (uint i = 0; i < iterations; ++i)
    {
        uint id = culled ? tile->indices[i] : i;
        Light light = lights.values[id];
        float3 L = light.position.xyz - v.worldPosition;
        float distance = length(L);
        float radius = light.position.w;
        float x = saturate(1.0f - distance / radius);
        float attenuation = x * x;
        if (attenuation > 0.0f)
        {
            float3 lDir = L / max(distance, 0.0001f);
            float diffuse = max(dot(n, lDir), 0.0f);
            float highlight = pow(max(dot(n, normalize(lDir + V)), 0.0f), 48.0f);
            color += light.color.rgb * (albedo * diffuse + specularStrength * highlight) * attenuation;
        }
    }
    return float4(color, 1.0f);
}

// Unlit light markers: one camera-facing billboard per point light.

struct UnlitVaryings
{
    float4 position : SV_Position;
    float3 color;
    float2 quadUV;
};

[shader("vertex")]
UnlitVaryings unlitVertex(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    float2 corners[6] = {
        float2(-1.0f, -1.0f),
        float2(1.0f, -1.0f),
        float2(1.0f, 1.0f),
        float2(-1.0f, -1.0f),
        float2(1.0f, 1.0f),
        float2(-1.0f, 1.0f),
    };
    Light light = lights.values[instanceID];
    float4 p = mul(view, float4(light.position.xyz, 1.0f));
    p.xy += corners[vertexID] * 0.14f;
    UnlitVaryings o;
    o.position = mul(projection, p);
    o.color = light.color.rgb;
    o.quadUV = corners[vertexID];
    return o;
}

[shader("fragment")]
float4 unlitMain(UnlitVaryings v)
    : SV_Target0
{
    if (dot(v.quadUV, v.quadUV) > 1.0f)
        discard;
    return float4(v.color * 1.5f, 1.0f);
}
