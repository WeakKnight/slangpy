// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tile shaders that exercise imageblock reads and persistent tile memory.
// `initialize` seeds both tile memory slots, `increment` / `incrementQuad` count the threads of each
// tile dispatch, and the fragment shader writes the accumulated values to the color attachment.

struct Pixel {
    float4 color : SV_Target0;
    float depth : SV_Target1;
};

// Tile memory slot 0, a 16-byte struct.
struct Counters {
    uint colorBits;
    uint threads;
    uint column;
    uint unused;
};

typealias CountersPtr = Ptr<Counters, Access.ReadWrite, AddressSpace.GroupShared>;
typealias Words = Ptr<uint, Access.ReadWrite, AddressSpace.GroupShared>;

// The imageblock sits in a uniform struct next to a constant that the host sets.
struct TileInputs {
    MetalImageBlock<Pixel> imageBlock;
    uint columnOffset;
};

uniform TileInputs tileInputs;

[shader("tile")]
[numthreads(16, 16, 1)]
void initialize(
    CountersPtr counters: SV_MetalThreadgroup0,
    uint index: SV_GroupIndex,
    uint3 group: SV_GroupID,
    Words aux: SV_MetalThreadgroup1
)
{
    if (index == 0) {
        counters->colorBits = asuint(tileInputs.imageBlock.read(uint2(0)).color.r);
        counters->threads = 0;
        counters->column = group.x + tileInputs.columnOffset;
        aux[0] = group.y;
        aux[1] = asuint(tileInputs.imageBlock.readSample(uint2(0), 0).depth);
    }
}

[shader("tile")]
[numthreads(16, 16, 1)]
void increment(CountersPtr counters: SV_MetalThreadgroup0)
{
    InterlockedAdd(counters->threads, 1);
}

[shader("tile")]
[numthreads(16, 16, 1)]
void incrementQuad(CountersPtr counters: SV_MetalThreadgroup0, uint index: SV_GroupIndex)
{
    uint value = 1;
    value += QuadReadAcrossX(value);
    value += QuadReadAcrossY(value);
    if (index == QuadReadLaneAt(index, 0))
        InterlockedAdd(counters->threads, value);
}

[shader("vertex")]
float4 vertexMain(uint id: SV_VertexID)
    : SV_Position
{
    return float4(id == 2 ? 3 : -1, id == 1 ? 3 : -1, 0, 1);
}

[shader("fragment")]
float4 fragmentMain(CountersPtr counters: SV_MetalThreadgroup0, Words aux: SV_MetalThreadgroup1)
    : SV_Target0
{
    return float4(
        asfloat(counters->colorBits) + asfloat(aux[1]),
        float(counters->threads) / 256,
        float(counters->column),
        float(aux[0]) + 1
    );
}
