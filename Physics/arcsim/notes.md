

lib arcsim 的笔记

```
template <typename T> T stp (const Vec<3,T> &u, const Vec<3,T> &v, const Vec<3,T> &w) {return dot(u,cross(v,w));}
```

find root，要求解的未知量是时间啊。

比如一个粒子，它的初始位置x0 ，速度为v0 。障碍物位置x1，障碍物速度是v1，要算它们什么时候会撞上，我们可以用显式方法，弄个循环慢慢迭代。

```python
x0 = 0
v0 = 10
x1 = 5
v1 = 0
dt = 0.01
steps = 1 / dt
for t in range(steps):
    x0 = x0 + dt * v0
    x1 = x1 + dt * v1
    if abs(x1 - x0) < 1e-10:
         collision = True
```

显式方法除了简单就没有任何优点了，而且只能在穿模和电脑算冒烟之间二选一。为了更快更准，我们可以算解析解
$$
\large
x_0 + v_0t = x_1 + v_1t
$$
只要t 小于给定的时间步dt，就可以认为是没撞上。这算得上是最简单的连续碰撞检测了，也叫做求根方法(root-finding method)。连续碰撞检测一下子就算出来了，甩出显式方法几条街。

不过解一元一次方程怎么过瘾呢？咱来玩点刺激的吧。连续碰撞检测主要可分为顶点-面(vertex-face)检测和边-边(edge-edge)检测。

三维
$$
F_{vf}(t,u,v) = p(t) - ((1-u-v)v_1(t) + uv_2(t) + vv_3(t))
$$
其中p是点的位置，v1v2v3是三角形三个顶点的位置。

也就是vertex-face case，经常在代码中简写为VF。



以上也可以看出，连续碰撞检测虽然比显式欧拉又快又准确，但还是很耗时间。所以在self-ccdhttps://gamma.cs.unc.edu/SELFCD/库以及arcsim库中，用的是层次包围盒BVH配合连续碰撞检测CCD检测的方式。如果层次包围盒在大场景粗略地检测到两个物体可能碰撞，再用连续碰撞检测精确计算究竟什么时候碰撞。

![image-20210807232601730](D:\图形学书籍\系列流体文章\gif\image-20210807232601730.png)

两个三角形的碰撞，可以拆分成6个vertex-face加上9个edge-edge。

vertex-face case 简写为VF
$$
F_{vf}(t,u,v) = p(t) - ((1-u-v)v_1(t) + uv_2(t) + vv_3(t))
$$
edge-edge case 简写为EE
$$
F_{ee}(t,u,v) = ((1-u)p_1(t) + up_2(t)) - ((1-v)p_3(t) + vp_4(t))
$$
