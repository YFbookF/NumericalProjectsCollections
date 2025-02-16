/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # https://github.com/NVIDIAGameWorks/Falcor
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
import Scene.SceneTypes;

/** Compute pass for skinned vertex animation.

    The dispatch size is one thread per dynamic vertex.
*/

struct SkinningData
{
    bool initPrev; ///< Copy current frame data to previous frame data.

    // Vertex data
    StructuredBuffer<PackedStaticVertexData> staticData;            ///< Original global vertex buffer. This holds the unmodified input vertices.
    StructuredBuffer<DynamicVertexData> dynamicData;                ///< Bone IDs and weights for all dynamic vertices.
    RWStructuredBuffer<PackedStaticVertexData> skinnedVertices;     ///< Skinned global vertex buffer. We'll update the positions only for the dynamic meshes.
    RWStructuredBuffer<PrevVertexData> prevSkinnedVertices;         ///< Previous frame vertex positions for dynamic meshes. Same size as 'dynamicData'.

    // Transforms
    StructuredBuffer<float4> boneMatrices;
    StructuredBuffer<float4> inverseTransposeBoneMatrices;
    StructuredBuffer<float4> worldMatrices;
    StructuredBuffer<float4> meshBindMatrices;
    StructuredBuffer<float4> meshInvBindMatrices;
    StructuredBuffer<float4> inverseTransposeWorldMatrices;

    // Accessors

    float4x4 getTransposeWorldMatrix(uint matrixID)
    {
        float4x4 m = float4x4(worldMatrices[matrixID * 4 + 0],
        worldMatrices[matrixID * 4 + 1],
        worldMatrices[matrixID * 4 + 2],
        worldMatrices[matrixID * 4 + 3]);

        return transpose(m);
    }

    float4x4 getInverseWorldMatrix(uint matrixID)
    {
        float4x4 m = float4x4(inverseTransposeWorldMatrices[matrixID * 4 + 0],
        inverseTransposeWorldMatrices[matrixID * 4 + 1],
        inverseTransposeWorldMatrices[matrixID * 4 + 2],
        inverseTransposeWorldMatrices[matrixID * 4 + 3]);

        return transpose(m);
    }

    float4x4 getMeshBindMatrix(uint matrixID)
    {
        return float4x4(meshBindMatrices[matrixID * 4 + 0],
        meshBindMatrices[matrixID * 4 + 1],
        meshBindMatrices[matrixID * 4 + 2],
        meshBindMatrices[matrixID * 4 + 3]);
    }

    float4x4 getInverseMeshBindMatrix(uint matrixID)
    {
        return float4x4(meshInvBindMatrices[matrixID * 4 + 0],
        meshInvBindMatrices[matrixID * 4 + 1],
        meshInvBindMatrices[matrixID * 4 + 2],
        meshInvBindMatrices[matrixID * 4 + 3]);
    }

    float4x4 getBoneMatrix(uint matrixID)
    {
        return float4x4(boneMatrices[matrixID * 4 + 0],
        boneMatrices[matrixID * 4 + 1],
        boneMatrices[matrixID * 4 + 2],
        boneMatrices[matrixID * 4 + 3]);
    }

    float4x4 getInverseTransposeBoneMatrix(uint matrixID)
    {
        return float4x4(inverseTransposeBoneMatrices[matrixID * 4 + 0],
        inverseTransposeBoneMatrices[matrixID * 4 + 1],
        inverseTransposeBoneMatrices[matrixID * 4 + 2],
        inverseTransposeBoneMatrices[matrixID * 4 + 3]);
    }

    float4x4 getBlendedMatrix(uint vertexId)
    {
        DynamicVertexData d = dynamicData[vertexId];

        float4x4 boneMat = getBoneMatrix(d.boneID.x) * d.boneWeight.x;
        boneMat += getBoneMatrix(d.boneID.y) * d.boneWeight.y;
        boneMat += getBoneMatrix(d.boneID.z) * d.boneWeight.z;
        boneMat += getBoneMatrix(d.boneID.w) * d.boneWeight.w;

        // Apply mesh bind transform before skinning (mesh to skeleton local at bind time)
        boneMat = mul(getMeshBindMatrix(d.bindMatrixID), boneMat);

        // Skinning takes us to world space, so apply the inverse to return to skeleton local
        boneMat = mul(boneMat, getInverseWorldMatrix(d.skeletonMatrixID));

        // Apply inverse bind matrix to return to mesh-local
        boneMat = mul(boneMat, getInverseMeshBindMatrix(d.bindMatrixID));
        return boneMat;
    }

    float4x4 getInverseTransposeBlendedMatrix(uint vertexId)
    {
        DynamicVertexData d = dynamicData[vertexId];

        float4x4 boneMat = getInverseTransposeBoneMatrix(d.boneID.x) * d.boneWeight.x;
        boneMat += getInverseTransposeBoneMatrix(d.boneID.y) * d.boneWeight.y;
        boneMat += getInverseTransposeBoneMatrix(d.boneID.z) * d.boneWeight.z;
        boneMat += getInverseTransposeBoneMatrix(d.boneID.w) * d.boneWeight.w;

        boneMat = mul(boneMat, getTransposeWorldMatrix(d.skeletonMatrixID));
        return boneMat;
    }

    uint getStaticVertexID(uint vertexId)
    {
        return dynamicData[vertexId].staticIndex;
    }

    StaticVertexData getStaticVertexData(uint vertexId)
    {
        return staticData[getStaticVertexID(vertexId)].unpack();
    }

    void storeSkinnedVertexData(uint vertexId, StaticVertexData data, PrevVertexData prevData)
    {
        gData.skinnedVertices[getStaticVertexID(vertexId)].pack(data);
        gData.prevSkinnedVertices[vertexId] = prevData;
    }

    float3 getCurrentPosition(uint vertexId)
    {
        return gData.skinnedVertices[getStaticVertexID(vertexId)].position;
    }
};

ParameterBlock<SkinningData> gData;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Check that this is an active vertex
    uint vertexId = dispatchThreadID.x;
    uint vertexCount, stride;
    gData.dynamicData.GetDimensions(vertexCount, stride);
    if (vertexId >= vertexCount) return;

    // Blend the vertices
    StaticVertexData s = gData.getStaticVertexData(vertexId);
    float4x4 boneMat = gData.getBlendedMatrix(vertexId);
    float4x4 invTransposeMat = gData.getInverseTransposeBlendedMatrix(vertexId);

    s.position = mul(float4(s.position, 1.f), boneMat).xyz;
    s.tangent.xyz = mul(s.tangent.xyz, (float3x3) boneMat);
    s.normal = mul(s.normal, (float3x3) transpose(invTransposeMat));

    // Copy the previous skinned data
    PrevVertexData prev;
    prev.position = gData.initPrev ? s.position : gData.getCurrentPosition(vertexId);

    // Store the result
    gData.storeSkinnedVertexData(vertexId, s, prev);
}
