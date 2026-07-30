// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

static const float PI = 3.14159265358979323846;
static const float INV_PI = 0.31830988618379067154;
static const float TWO_PI = 6.28318530717958647692;

static const float3 LIGHT_CENTER = float3(0.0, 1.985, 1.0);
static const float3 LIGHT_HALF_U = float3(0.234, 0.0, 0.0);
static const float3 LIGHT_HALF_V = float3(0.0, 0.0, 0.188);
static const float3 LIGHT_NORMAL = float3(0.0, -1.0, 0.0);
static const float3 LIGHT_RADIANCE = float3(30.0, 25.0, 18.0);
static const float LIGHT_AREA = 0.175968;


struct Ray
{
    float3 origin;
    float3 direction;
    float t_min;
    float t_max;

    RayDesc to_ray_desc()
    {
        return {origin, t_min, direction, t_max};
    }
};


struct RNG
{
    uint state;

    __init(uint2 pixel, uint sample_index, uint seed)
    {
        state = (pixel.x * 100591u) ^ (pixel.y * 102191u) ^ (sample_index * 103643u) ^ seed;
        state ^= 0x9e3779b9u;
    }

    [mutating]
    uint next_uint()
    {
        state = 1664525u * state + 1013904223u;
        return state;
    }

    [mutating]
    float next_float()
    {
        return (next_uint() >> 8) * 0x1p-24;
    }

    [mutating]
    float2 next_float2()
    {
        return float2(next_float(), next_float());
    }
};


struct Scene
{
    RaytracingAccelerationStructure tlas;
    StructuredBuffer<float3> vertices;
    StructuredBuffer<uint> indices;
    StructuredBuffer<uint2> instance_mesh_ranges;
    StructuredBuffer<float4x4> instance_transforms;
    StructuredBuffer<float4> instance_materials;
};


struct Hit
{
    bool valid;
    float3 position;
    float3 normal;
    float3 albedo;
    float emission;
};


ParameterBlock<Scene> g_scene;
RWTexture2D<float4> g_accumulation;
Texture2D<float4> g_accumulation_input;
RWTexture2D<float4> g_output;
uniform uint g_sample_offset;
uniform uint g_samples_per_pass;
uniform uint g_max_depth;
uniform uint g_seed;
uniform bool g_reset_accumulation;


Hit trace_closest(Ray ray)
{
    Hit hit = {};
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(g_scene.tlas, 0, 0xff, ray.to_ray_desc());
    while (query.Proceed())
    {
    }

    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return hit;

    uint instance_index = query.CommittedInstanceIndex();
    uint primitive_index = query.CommittedPrimitiveIndex();
    float2 barycentrics = query.CommittedTriangleBarycentrics();
    uint2 mesh_range = g_scene.instance_mesh_ranges[instance_index];
    uint index_offset = mesh_range.y + primitive_index * 3;
    uint i0 = mesh_range.x + g_scene.indices[index_offset + 0];
    uint i1 = mesh_range.x + g_scene.indices[index_offset + 1];
    uint i2 = mesh_range.x + g_scene.indices[index_offset + 2];
    float4x4 transform = g_scene.instance_transforms[instance_index];
    float3 p0 = mul(transform, float4(g_scene.vertices[i0], 1.0)).xyz;
    float3 p1 = mul(transform, float4(g_scene.vertices[i1], 1.0)).xyz;
    float3 p2 = mul(transform, float4(g_scene.vertices[i2], 1.0)).xyz;
    float3 bary = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics);
    float3 normal = normalize(cross(p1 - p0, p2 - p0));
    if (dot(normal, ray.direction) > 0.0)
        normal = -normal;
    float4 material = g_scene.instance_materials[instance_index];

    hit.valid = true;
    hit.position = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    hit.normal = normal;
    hit.albedo = material.xyz;
    hit.emission = material.w;
    return hit;
}


bool trace_shadow(float3 origin, float3 direction, float distance)
{
    Ray ray = {origin, direction, 1e-4, max(distance - 2e-4, 1e-4)};
    RayQuery<
        RAY_FLAG_FORCE_OPAQUE |
        RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>
        query;
    query.TraceRayInline(g_scene.tlas, 0, 0xff, ray.to_ray_desc());
    while (query.Proceed())
    {
    }
    return query.CommittedStatus() == COMMITTED_NOTHING;
}


float3 sample_cosine_hemisphere(float3 normal, float2 sample)
{
    float radius = sqrt(sample.x);
    float phi = TWO_PI * sample.y;
    float3 local_direction =
        float3(radius * cos(phi), radius * sin(phi), sqrt(max(0.0, 1.0 - sample.x)));
    float3 tangent =
        normalize(abs(normal.z) < 0.999 ? cross(float3(0.0, 0.0, 1.0), normal)
                                       : cross(float3(0.0, 1.0, 0.0), normal));
    float3 bitangent = cross(normal, tangent);
    return normalize(
        tangent * local_direction.x + bitangent * local_direction.y + normal * local_direction.z
    );
}


float3 sample_direct_light(Hit hit, inout RNG rng)
{
    float2 light_sample = rng.next_float2() * 2.0 - 1.0;
    float3 light_position =
        LIGHT_CENTER + LIGHT_HALF_U * light_sample.x + LIGHT_HALF_V * light_sample.y;
    float3 to_light = light_position - hit.position;
    float distance_squared = dot(to_light, to_light);
    float distance = sqrt(distance_squared);
    float3 light_direction = to_light / distance;
    float surface_cosine = max(0.0, dot(hit.normal, light_direction));
    float light_cosine = max(0.0, dot(LIGHT_NORMAL, -light_direction));
    if (surface_cosine == 0.0 || light_cosine == 0.0)
        return float3(0.0);
    if (!trace_shadow(hit.position + hit.normal * 1e-4, light_direction, distance))
        return float3(0.0);
    float geometry_term = surface_cosine * light_cosine / distance_squared;
    return hit.albedo * INV_PI * LIGHT_RADIANCE * geometry_term * LIGHT_AREA;
}


Ray make_camera_ray(uint2 pixel, uint2 dimensions, float2 jitter)
{
    float2 uv = (float2(pixel) + jitter) / float2(dimensions);
    float aspect_ratio = float(dimensions.x) / float(dimensions.y);
    float tan_half_fov = tan(19.65385 * PI / 180.0);
    float2 screen = float2(
        (uv.x * 2.0 - 1.0) * tan_half_fov,
        (1.0 - uv.y * 2.0) * tan_half_fov / aspect_ratio
    );
    float3 origin = float3(0.0, 0.995, -2.861);
    float3 direction = normalize(float3(screen.x, screen.y, 1.0));
    return {origin, direction, 0.0, 1000.0};
}


float3 trace_path(uint2 pixel, uint2 dimensions, uint sample_index)
{
    RNG rng = RNG(pixel, sample_index, g_seed);
    Ray ray = make_camera_ray(pixel, dimensions, rng.next_float2());
    float3 radiance = float3(0.0);
    float3 throughput = float3(1.0);

    for (uint depth = 0; depth < g_max_depth; ++depth)
    {
        Hit hit = trace_closest(ray);
        if (!hit.valid)
            break;

        if (hit.emission > 0.0)
        {
            if (depth == 0)
                radiance += throughput * LIGHT_RADIANCE * hit.emission;
            break;
        }

        radiance += throughput * sample_direct_light(hit, rng);
        throughput *= hit.albedo;

        if (depth >= 2)
        {
            float continuation_probability =
                clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.95);
            if (rng.next_float() > continuation_probability)
                break;
            throughput /= continuation_probability;
        }

        float3 next_direction = sample_cosine_hemisphere(hit.normal, rng.next_float2());
        ray = {
            hit.position + hit.normal * 1e-4,
            next_direction,
            0.0,
            1000.0,
        };
    }

    return radiance;
}


[shader("compute")]
[numthreads(16, 16, 1)]
void trace_paths(uint3 tid: SV_DispatchThreadID)
{
    uint2 dimensions;
    g_accumulation.GetDimensions(dimensions.x, dimensions.y);
    uint2 pixel = tid.xy;
    if (any(pixel >= dimensions))
        return;

    float3 radiance = float3(0.0);
    for (uint sample = 0; sample < g_samples_per_pass; ++sample)
        radiance += trace_path(pixel, dimensions, g_sample_offset + sample);

    float4 accumulated =
        g_reset_accumulation ? float4(0.0) : g_accumulation[pixel];
    accumulated += float4(radiance, float(g_samples_per_pass));
    g_accumulation[pixel] = accumulated;
}


float3 aces_film(float3 color)
{
    color *= 0.6;
    return saturate(
        (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14)
    );
}


[shader("compute")]
[numthreads(16, 16, 1)]
void tone_map(uint3 tid: SV_DispatchThreadID)
{
    uint2 dimensions;
    g_output.GetDimensions(dimensions.x, dimensions.y);
    uint2 pixel = tid.xy;
    if (any(pixel >= dimensions))
        return;

    float4 accumulated = g_accumulation_input.Load(int3(pixel, 0));
    float3 color = accumulated.xyz / max(accumulated.w, 1.0);
    g_output[pixel] = float4(aces_film(color), 1.0);
}
