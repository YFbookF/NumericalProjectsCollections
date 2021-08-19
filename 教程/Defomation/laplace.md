http://www.alecrivers.com/fastlsm/

首先已删除构造函数的错误，把SumData中的结构体修改为如下形式即可。吐个槽，我用别人电脑的时候，只报“引用已删除函数”的错误。我找了半天也没找到错误之处。然后用自己电脑，vs2019还添了一条”Vector3f包含不常用的构造函数“警告，才找到错误之处。

```
__declspec(align(16)) struct SumData
{
public:
	union {
		// Floats
		struct {
			__declspec(align(16)) Vector3f v;
			__declspec(align(16)) Matrix3x3f M;
		};
		struct {
			__m128 m1, m2, m3;
		};
	};

	SumData()
	{
		v = { 0,0,0 };
	}
};

```

然后是系统找不到exe，把输出的目录下的Testbed_d.exe改名为Testbed.exe即可。

这个程序运行起来如下

![image-20210818221855695](D:\图形学书籍\系列流体文章\gif\image-20210818221855695.png)

感觉挺像混合网格例子方法。

https://github.com/danielflower/MeshlessDeformations 超级棒，有一篇Siggraph Courses

对于一堆粒子，它原来在x0，现在在xn，经过两个变换，t和t后，需要找到一个旋转矩阵，能够最小化下面的算术
$$
\sum_i w_i (\bold R (\bold x_i^0 - \bold t_0) + \bold t - \bold x_i)^2
$$
我们可以这样计算
$$
\bold q_i = \bold x_i^0 - \bold x_{cm}^0 \qquad \bold p_i = \bold x_i - \bold x_{cm}
$$
那么上面的的式子可以重写如下
$$
\sum_i m_i(\bold A \bold q_i - \bold p_i)^2
$$
为了让上面这个式子最小化，解即为
$$
\bold A = (\sum_i m_i \bold p_i \bold q_i^T)(\sum_i m_i \bold q_i \bold q_i^T)^{-1}
$$
然后就可以用极分解来计算了
$$
\bold S = \sqrt{\bold A^T \bold A} \qquad \bold R = \bold A \bold S^{-1}
$$
注意原论文那么方程
$$
v(t+h) = v(t) + h\frac{-k(x(t) - l_0)}{m} \\
x(t+h) = x(t) + hv(t+h)
$$
可以换成下面的方程
$$
A = \begin{bmatrix} 1 & -kh/m \\ h & 1- h^2k/m\end{bmatrix}
$$
特征值
$$
e_0 = 1 - \frac{h^2k - \sqrt{-4mh^2k + h^4k^2}}{2m}
$$
和特征值
$$
e_1 = 1 - \frac{h^2k + \sqrt{4mh^2k + h^4k^2}}{2m}
$$
