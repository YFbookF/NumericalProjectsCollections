```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AtocmicManager : MonoBehaviour
{
    // Start is called before the first frame update
    ComputeBuffer test_num;
    public ComputeShader _shader;
    int kernel;
    int[] num = { 0 };

    int buffer_size = 8;
    int[] pos = { 1, 2, 3, 4, 5, 6, 7, 8 };
    ComputeBuffer args_buffer;
    uint[] args = new uint[5] { 0, 0, 0, 0, 0 };

    ComputeBuffer Particle_Buffer;
    [SerializeField] Mesh instance_mesh;
    [SerializeField] Material instance_material;
    void Start()
    {
        test_num = new ComputeBuffer(1,sizeof(int));
        test_num.SetData(num);
        kernel = _shader.FindKernel("CSMain");
        _shader.SetBuffer(kernel,"test", test_num);
        _shader.Dispatch(kernel, 64, 1, 1);
        test_num.GetData(num);

        Particle_Buffer = new ComputeBuffer(buffer_size, sizeof(int));
        Particle_Buffer.SetData(pos);

        args_buffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        uint numIndices = (uint)instance_mesh.GetIndexCount(0);
        args[0] = numIndices;
        args[1] = (uint)Particle_Buffer.count;
        args_buffer.SetData(args);

        instance_material.SetBuffer("particle_buffer", Particle_Buffer);
    }
    
    // Update is called once per frame
    void Update()
    {
        Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 100, 100));
        Graphics.DrawMeshInstancedIndirect(instance_mesh, 0, instance_material, bounds, args_buffer);
    }
}

```

