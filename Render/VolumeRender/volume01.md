https://github.com/SlightlyMad/VolumetricLights/tree/b29aff07336872a2f192d29f55d08b706ddffc54

首先定义射线ray，以及世界坐标。修改v2f结构体

```
struct v2f
			{
				float4 pos : SV_POSITION;
				float4 uv : TEXCOORD0;
				float3 ray : TEXCOORD1;
				float3 wpos : TEXCOORD2;
			};

			v2f vert(appdata v)
			{
				v2f o;
				o.pos = mul(_WorldViewProj, v.vertex);
//_material.SetMatrix("_WorldViewProj", viewProj * world);
				o.uv = ComputeScreenPos(o.pos);
				o.ray = mul(_WorldView, v.vertex).xyz * float3(-1, -1, 1);
				o.wpos = mul(unity_ObjectToWorld, v.vertex);
				return o;
			}
```

viewspace屏幕空间

https://qiita.com/yuji_yasuhara/items/98e6bd84b82e666496ce

Matrix4x4 proj = _camera.projectionMatrix;
$$
\begin{bmatrix} 0 & 0 & 0 & 0 \\0 & 1/tan(fov/2*\pi/180) & 0 & 0 \\0 & 0 & -\frac{far + near}{far - near} & -\frac{2far * near}{far - near} \\0 & 0 & -1 & 0 \end{bmatrix}
$$
