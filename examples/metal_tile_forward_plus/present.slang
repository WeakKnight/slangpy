// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tone-maps the floating-point Forward+ output onto the window's swapchain image.

Texture2D<float4> source;
SamplerState sourceSampler;
uint2 outputSize;

[shader("vertex")]
float4 vertexMain(uint id: SV_VertexID)
    : SV_Position
{
    return float4(id == 2 ? 3 : -1, id == 1 ? 3 : -1, 0, 1);
}

[shader("fragment")]
float4 fragmentMain(float4 position: SV_Position)
    : SV_Target0
{
    float3 color = max(source.SampleLevel(sourceSampler, position.xy / float2(outputSize), 0).rgb, 0);
    color /= 1 + color;
    color = select(color <= 0.0031308f, color * 12.92f, 1.055f * pow(color, 1.0f / 2.4f) - 0.055f);
    return float4(color, 1);
}
