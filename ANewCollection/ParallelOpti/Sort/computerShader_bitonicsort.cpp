https://github.com/kodai100/Unity_GPUNearestNeighbor/blob/master/Assets/Packages/NeghborSearchCS/Src/Common/Resources/BitonicSort.compute
﻿//--------------------------------------------------------------------------------------
// Pragmas
//--------------------------------------------------------------------------------------

#pragma kernel BitonicSort
#pragma kernel MatrixTranspose

//--------------------------------------------------------------------------------------
// Constants
//--------------------------------------------------------------------------------------

#define BITONIC_BLOCK_SIZE 512
#define TRANSPOSE_BLOCK_SIZE 16

//--------------------------------------------------------------------------------------
// Constant Buffers
//--------------------------------------------------------------------------------------
cbuffer CB
{
	uint _Level;
	uint _LevelMask;
	uint _Width;
	uint _Height;
};

//--------------------------------------------------------------------------------------
// Structured Buffers
//--------------------------------------------------------------------------------------
StructuredBuffer  <uint2> Input : register(t0);
RWStructuredBuffer<uint2> Data  : register(u0);

//--------------------------------------------------------------------------------------
// Bitonic Sort Compute Shader
//--------------------------------------------------------------------------------------
groupshared uint2 shared_data[BITONIC_BLOCK_SIZE];

bool Compare(uint2 left, uint2 right) {
	return (left.x == right.x) ? (left.y <= right.y) : (left.x <= right.x);
	// return left.x <= right.x;
}

[numthreads(BITONIC_BLOCK_SIZE, 1, 1)]
void BitonicSort(uint3 Gid  : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint  GI : SV_GroupIndex) {
	// Load shared data
	shared_data[GI] = Data[DTid.x];
	GroupMemoryBarrierWithGroupSync();

	// Sort the shared data
	for (uint j = _Level >> 1; j > 0; j >>= 1) {
		uint2 result = (Compare(shared_data[GI & ~j], shared_data[GI | j]) == (bool)(_LevelMask & DTid.x)) ? shared_data[GI ^ j] : shared_data[GI];
		GroupMemoryBarrierWithGroupSync();
		shared_data[GI] = result;
		GroupMemoryBarrierWithGroupSync();
	}

	// Store shared data
	Data[DTid.x] = shared_data[GI];
}

//--------------------------------------------------------------------------------------
// Matrix Transpose Compute Shader
//--------------------------------------------------------------------------------------
groupshared uint2 transpose_shared_data[TRANSPOSE_BLOCK_SIZE * TRANSPOSE_BLOCK_SIZE];

[numthreads(TRANSPOSE_BLOCK_SIZE, TRANSPOSE_BLOCK_SIZE, 1)]
void MatrixTranspose(uint3 Gid  : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint  GI : SV_GroupIndex) {
	transpose_shared_data[GI] = Input[DTid.y * _Width + DTid.x];
	GroupMemoryBarrierWithGroupSync();
	uint2 XY = DTid.yx - GTid.yx + GTid.xy;
	Data[XY.y * _Height + XY.x] = transpose_shared_data[GTid.x * TRANSPOSE_BLOCK_SIZE + GTid.y];
}