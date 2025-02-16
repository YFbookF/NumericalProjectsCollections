https://github.com/kodai100/Unity_GPUNearestNeighbor/blob/master/Assets/Packages/NeghborSearchCS/Src/3D/Shaders/Resources/Grid3D.compute
#pragma kernel BuildGridCS
#pragma kernel ClearGridIndicesCS
#pragma kernel BuildGridIndicesCS
#pragma kernel RearrangeParticlesCS
#pragma kernel CopyBuffer

#define SIMULATION_BLOCK_SIZE 32

#include "MyStructData.cginc"

StructuredBuffer<MyData>	_ParticlesBufferRead;
RWStructuredBuffer<MyData>	_ParticlesBufferWrite;

StructuredBuffer  <uint2>	_GridBufferRead;
RWStructuredBuffer<uint2>	_GridBufferWrite;

StructuredBuffer  <uint2>	_GridIndicesBufferRead;
RWStructuredBuffer<uint2>	_GridIndicesBufferWrite;

int _NumParticles;
float3 _GridDim;
float _GridH;

// -----------------------------------------------------------
// Grid
// -----------------------------------------------------------

// 所属するセルの2次元インデックスを返す
float3 GridCalculateCell(float3 pos) {
	return pos / _GridH;
}

// セルの2次元インデックスから1次元インデックスを返す
uint GridKey(uint3 xyz) {
	return xyz.x + xyz.y * _GridDim.x + xyz.z * _GridDim.x * _GridDim.y;
}

// (グリッドID, パーティクルID) のペアを作成する
uint2 MakeKeyValuePair(uint3 xyz, uint value) {
	// uint2([GridHash], [ParticleID]) 
	return uint2(GridKey(xyz), value);	// 逆?
}

// グリッドIDとパーティクルIDのペアからグリッドIDだけを抜き出す
uint GridGetKey(uint2 pair) {
	return pair.x;
}

// グリッドIDとパーティクルIDのペアからパーティクルIDだけを抜き出す
uint GridGetValue(uint2 pair) {
	return pair.y;
}

//--------------------------------------------------------------------------------------
// Build Grid : 各パーティクルの属するセルを計算し、紐づけてGridBufferに保存 -> 確認済み
//--------------------------------------------------------------------------------------
[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void BuildGridCS(uint3 DTid : SV_DispatchThreadID) {
	const unsigned int P_ID = DTid.x;	// Particle ID to operate on

	float3 position = _ParticlesBufferRead[P_ID].pos;
	float3 grid_xyz = GridCalculateCell(position);

	_GridBufferWrite[P_ID] = MakeKeyValuePair((uint3)grid_xyz, P_ID);
}

//--------------------------------------------------------------------------------------
// Build Grid Indices : ソート済みのパーティクルハッシュに対して、始まりと終わりを記録 -> 要確認
//--------------------------------------------------------------------------------------
// 0000011111122222334444 を
//       0 1  2  3  4
// start 0 5  11 16 18
// end   4 10 15 17 21
// に変換

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void ClearGridIndicesCS(uint3 DTid : SV_DispatchThreadID) {
	// グリッドの個数分
	_GridIndicesBufferWrite[DTid.x] = uint2(0, 0);
}

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void BuildGridIndicesCS(uint3 DTid : SV_DispatchThreadID) {
	// パーティクルの個数分
	const unsigned int P_ID = DTid.x;

	// 1個前のパーティクルIDを計算
	uint P_ID_PREV = (P_ID == 0) ? (uint)_NumParticles : P_ID;
	P_ID_PREV--;

	// 1個後のパーティクルIDを計算
	uint P_ID_NEXT = P_ID + 1;
	if (P_ID_NEXT == (uint)_NumParticles) { P_ID_NEXT = 0; }

	// ソート済みのGrid-Particleバッファから
	// 自分がいるグリッドを計算する
	uint cell = GridGetKey(_GridBufferRead[P_ID]);				// ソートされたグリッドIDの取得
	uint cell_prev = GridGetKey(_GridBufferRead[P_ID_PREV]);
	uint cell_next = GridGetKey(_GridBufferRead[P_ID_NEXT]);

	// 前後セルインデックスと異なる場合記録
	if (cell != cell_prev) {
		// 新しいセルインデックスの始まりの配列インデックス
		_GridIndicesBufferWrite[cell].x = P_ID;
	}

	if (cell != cell_next) {
		// 新しいセルインデックスの終わりの配列インデックス
		_GridIndicesBufferWrite[cell].y = P_ID + 1;
	}
}

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void RearrangeParticlesCS(uint3 DTid : SV_DispatchThreadID) {
	const unsigned int id = DTid.x; // Particle ID to operate on
	const unsigned int P_ID = GridGetValue(_GridBufferRead[id]);
	_ParticlesBufferWrite[id] = _ParticlesBufferRead[P_ID];	// ソート済みに並び替える
}

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void CopyBuffer(uint3 DTid : SV_DispatchThreadID) {
	uint id = DTid.x;
	_ParticlesBufferWrite[id] = _ParticlesBufferRead[id];
}
