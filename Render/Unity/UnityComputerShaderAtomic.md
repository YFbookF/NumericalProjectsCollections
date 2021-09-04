

```
    ComputeBuffer test_num;
    public ComputeShader _shader;
    int kernel;
    int[] num = { 0 };
    void Start()
    {
        test_num = new ComputeBuffer(1,sizeof(int));
        test_num.SetData(num);
        kernel = _shader.FindKernel("CSMain");
        _shader.SetBuffer(kernel,"test", test_num);
        _shader.Dispatch(kernel, 64, 1, 1);
        test_num.GetData(num);
        Debug.Log(num[0]);
    }
```

计算着色器这么写就行了

```
#pragma kernel CSMain

RWStructuredBuffer<int> test;

[numthreads(8,1,1)]
void CSMain (uint3 id : SV_DispatchThreadID)
{

    InterlockedAdd(test[0], 1);
}

```

