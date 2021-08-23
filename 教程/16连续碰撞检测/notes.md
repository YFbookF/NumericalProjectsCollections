### 【游戏物理笔记】连续碰撞检测

### 一点感想

我最近终于意识到，如果一本有关图形学或计算物理的书籍或论文，附上代码的话，那么阅读难度会直接下降90%。而如果还能把代码运行起来的话，那么难度会下降99%。之前我总想着看完资料就能找到相应的参考代码，但翻遍github也找不到。但如果反过来，先看代码再找相应的资料，那就很容易了，效率也高。

但即使这样，代码仍然不容易寻找。除了github外，我参考代码的主要来源，就是各教授在自己主页上为自己的publications所附的代码了。虽然不是每篇都有，甚至大多数代码都是论文实现的简化版，但这些散落在互联网各地的代码帮助仍然很大。

所以这个专栏会继续它的目标，那就是收集这些散落在互联网各地的代码，以及介绍一些不错的库，专挑那些几乎没被提及的很难寻找的代码。但这些也得益于互联网的开源精神，因为有了那些开放了自己的代码的先锋，我才能躺自己家里沙发上，同时去学习世界上最优秀的代码。

收集代码的范围就包括一切有关物理模拟的，反正什么计算力学，计算流体力学，图形学很多算法都能通用，干脆都塞进来。而且“你那个那么好玩，我得整一个更好”。序号就不编了，因为我也不知道我能遇上什么样的代码。

### 连续碰撞检测

比如一个一维空间中，一个点所在位置x0，它的速度v0。第二个点所在位置x1，它的速度v1。那么第一个点什么时候能遇到第二个点？

我们可以用显式方法，先选一个很小的时间步，然后慢慢迭代

```
x0 = 0
v0 = 10
x1 = 5
v1 = 0
dt = 0.2
steps = 1 / dt
for t in range(steps):
    x0 = x0 + dt * v0
    x1 = x1 + dt * v1
    if abs(x1 - x0) < 1e-10:
         collision = True
```

很简单也很没效率，并且接下来就要在穿模和电脑算冒烟直接二选一了。但是我们也可以直接列方程
$$
x_0 + v_0t = x_1 + v_1t 
$$
它的解析解如下
$$
\\
t = \frac{x_1 - x_0}{v_0 - v_1}
$$
这就是最简单的连续碰撞检测了。注意，在这个以及之后方程中，时间t才是我们要求解的那个未知量。

### 二维

好了，现在快进到二维的。首先是顶点-边(vetex-edge)检测。如果一个顶点恰好在一个边上，那么根据中学学过的知识
$$
F(u,t) = (\bold P_0(t) - \bold P_1(t)) - u(\bold P_2(t) - \bold P_1(t)) = 0
$$
其中P0是点的位置，P1P2是边上顶点的位置。一个方程两个未知量，算起来太麻烦，来人，把上面这个公式拖下去斩了，给朕换个公式
$$
F(t) = (\bold P_0(t) - \bold P_1(t)) \times (\bold P_2(t) - \bold P_1(t)) = 0
$$
这不挺好，一个未知量就方便搞事了。接下来把t弄出来
$$
((\bold x_0 + t \bold v_0) - (\bold x_1 + t \bold v_1))\times ((\bold x_2 + t \bold v_2) - (\bold x_1 + t \bold v_1)) = 0
$$
那么可以化成一个一元二次方程
$$
at^2 + bt + c = 0
$$
其中
$$
a = (\bold v_0 - \bold v_1) \times(\bold v_2 - \bold v_1)\\
b = (\bold x_0 - \bold x_1) \times(\bold v_2 - \bold v_1)  + (\bold v_0 - \bold v_1) \times(\bold x_2 - \bold x_1) \\
c = (\bold x_0 - \bold x_1) \times(\bold x_2 - \bold x_1)
$$
上面这些公式写成python代码如下

```python
import numpy as np
x0 = np.array([0.0,0.0]) # 点的位置
x1 = np.array([1.0,0.0]) # 边的顶点1的位置
x2 = np.array([0.0,1.0]) # 边的顶点2的位置
v0 = np.array([1.0,1.0]) # 点的速度
v1 = np.array([0.0,0.0]) # 边的顶点1的速度
v2 = np.array([0.0,0.0]) # 边的顶点2的速度

def cross2d(vec0,vec1):
    return vec0[0]*vec1[1] - vec0[1]*vec1[0]

a = cross2d(v0-v1, v2-v1)
d = cross2d(x0-x1, v2-v1) + cross2d(v0-v1, x2-x1)
c = cross2d(x0-x1, x2-x1)
```

接下来就是怎么解这个一元二次方程了。不过为了减小误差，根据wiki，可以得出下面你没写过的全新版本：

```
d = b*b - 4*a*c
result = np.zeros((2))
result_num = 0
if d < 0:
    result_num = 1
    result[1] = - b / (2 * a)
else:
    q = - (b + np.sign(b)*np.sqrt(d)) / 2
    if (abs(a) > 1e-12*abs(q)):
        result[result_num] = q / a
        result_num += 1
    if (abs(q) > 1e-12*abs(c)):
        result[result_num] = c / q
        result_num += 1
    if result_num == 2 and result[0] > result[1]:
        temp = result[0]
        result[0] = result[1]
        result[1] = temp
```

我可不是瞎写，因为在berkeley图形学实验室上开源的arcsim库中的util.cpp中也是这么写的。http://graphics.berkeley.edu/resources/ARCSim/。

arcsim是用于一个用于模拟形变材料的库，目前版本是0.3.1，最后一次更新是2014年之后。除了连续碰撞检测外，这个库还有其它代码很值得一看。此篇文章也是因为有这个开源库才得以写出。由于我认为这份代码并不好找，所以也将这份代码收录进了此篇文章的仓库中。

最后还有一个坑，就算把一元二次的解算出来了，这个解也不一定有物理含义，所以最后还要判断在那个时刻，点是不是真的在线段上。 

```
collision = False
for i in range(result_num):
    t = result[i]
    x0new = x0 + t * v0
    x1new = x1 + t * v1
    x2new = x2 + t * v2
    res = cross2d(x0new - x1new, x2new - x1new)
    if abs(res) < 1e-10:
        collision = True
```

### 三维

三维连续碰撞检测包括顶点-面(vertex - face)和边-边(edge-edge)。

不过对于顶点-面来说，大多数时候就是顶点-三角形。先列方程。
$$
F(t) = (\bold P_0(t) - \bold P_1(t)) \cdot ((\bold P_2(t) - \bold P_1(t)) \times(\bold P_3(t) - \bold P_1(t))) = 0
$$
其中P0就是顶点，P1P2P3就是三角形三个点。含义就是三角形平面上的一个向量，肯定和三角形法向量垂直，也就是点积结果为零。而三角形法向量则由两条边的向量叉积求得。继续变换如下
$$
((\bold x_0 + t \bold v_0) - (\bold x_1 + t \bold v_1))\cdot (((\bold x_2 + t \bold v_2) - (\bold x_1 + t \bold v_1)) \\\times ((\bold x_3 + t \bold v_3) - (\bold x_1 + t \bold v_1))) = 0
$$
那么就得了下面的一元三次方程
$$
at^3 + bt^2 + ct  +d = 0
$$
其中
$$
a = \bold v_{01} \cdot (\bold v_{21} \times \bold v_{31}) \\
b = \bold x_{01} \cdot (\bold v_{21} \times \bold v_{31}) + \bold v_{01} \cdot (\bold v_{21} \times \bold x_{31})  +\bold v_{01} \cdot (\bold x_{21} \times \bold v_{31})  \\
c = \bold x_{01} \cdot (\bold v_{21} \times \bold x_{31}) + \bold x_{01} \cdot (\bold x_{21} \times \bold v_{31})  +\bold v_{01} \cdot (\bold x_{21} \times \bold x_{31})  \\
d = \bold x_{01} \cdot (\bold x_{21} \times \bold x_{31})
$$
写成python代码如下

```python
pos0 = np.array([1,1,0]) # 点P的位置 
pos1 = np.array([0,0,1]) # 三角形顶点 1 的位置
pos2 = np.array([1,0,1]) # 三角形顶点 2 的位置
pos3 = np.array([0,1,1]) # 三角形顶点 3 的位置

vel0 = np.array([0,0,0]) # 点 P 的速度
vel1 = np.array([0,0,-1]) # 三角形顶点 1 的速度
vel2 = np.array([-1,1,-1]) # 三角形顶点 2 的速度
vel3 = np.array([1,-1,-1]) # 三角形顶点 3 的速度

v01 = vel0 - vel1
v21 = vel2 - vel1
v31 = vel3 - vel1

x01 = pos0 - pos1
x21 = pos2 - pos1
x31 = pos3 - pos1

coeff_a = dot(v01,cross(v21,v31))
coeff_b = dot(x01,cross(v21,v31)) + dot(v01,cross(v21,x31)) + dot(v01,cross(x21,v31))
coeff_c = dot(x01,cross(v21,x31)) + dot(x01,cross(x21,v31)) + dot(v01,cross(x21,x31))
coeff_d = dot(x01,cross(x21,x31))
```

怎么解这个一元三次方程之后再说。现在先看看如何处理边边的情况。先列方程
$$
F(t) = (\bold P_2(t) - \bold P_0(t)) \cdot ((\bold P_1(t) - \bold P_0(t)) \times(\bold P_3(t) - \bold P_2(t))) = 0
$$
其中P0P1是第一条边的两个点，P2P3是第二条边的两个点。含义就是如果两条边碰撞了，那么叉积两个边，得出两个边所在平面法向量，然后再找一个所在平面的一个向量，点积结果肯定是零。化为如下式子
$$
((\bold x_2 + t \bold v_2) - (\bold x_0 + t \bold v_0))\cdot (((\bold x_1 + t \bold v_1) - (\bold x_0 + t \bold v_0)) \\\times ((\bold x_3 + t \bold v_3) - (\bold x_2 + t \bold v_2))) = 0
$$
那么一元三次方程的参数如下
$$
a = \bold v_{20} \cdot (\bold v_{10} \times \bold v_{32}) \\
b = \bold x_{20} \cdot (\bold v_{10} \times \bold v_{32}) + \bold v_{20} \cdot (\bold v_{10} \times \bold x_{32})  +\bold v_{20} \cdot (\bold x_{10} \times \bold v_{32})  \\
c = \bold x_{20} \cdot (\bold v_{10} \times \bold x_{32}) + \bold x_{20} \cdot (\bold x_{10} \times \bold v_{32})  +\bold v_{20} \cdot (\bold x_{10} \times \bold x_{32})  \\
d = \bold x_{20} \cdot (\bold x_{10} \times \bold x_{32})
$$
代码和之前写法差不多。

### 一元三次方程解法

由于a,b,c,d可能是任意数，所以上面的一元三次方程并没有解析解。咋办呢？解这种一元三次方程，常用的方法包括牛顿迭代法，二分法，以及牛顿迭代法与二分法一起使用。

最简单的二分法，使用的库有lostops https://github.com/mtao/LosTopos/blob/41928608983331a597de557867cb5cecc8d90c09/src/common/cubic_ccd_wrapper.cpp#L383。

牛顿迭代法好歹也是解非线性方程的老大哥了，不认识一下怎么行。目标为找到一个x，使得F(x) = 0。公式如下
$$
\bold x^{k+1} = \bold x^{k} - \frac{F(\bold x^k)}{F'(\bold x^k)}
$$
其含义就是，每次走一点，当F(x) = 0，说明找到根了，那么停止迭代。还要除个斜率的原因是，如果斜率过大，那么走慢一些，防止走过头。如果斜率过小，那就别磨磨唧唧了，那么走快一点。写成代码如下。

```python
import numpy as np
a = 1
b = 1
c = 1
d = 1
x = 10
for ite in range(100):
    f = a*x*x*x + b*x*x + c*x + d
    df = 3*a*x*x + b*x + c
    s = f / df
    if abs(s) < 1e-10:
        break
    x = x - f / df
```

不过直接用牛顿迭代法是不行的，因为一元三次方程最多可能有三个根，而牛顿迭代法只能找一个。

解决这个问题的方法之一，就是先用极值把一元三次方程分成三段，比如[0,x1]以及[x1,x2]以及[x2,1]三段。然后分别用牛顿迭代法求。一元三次方程的极值处就它斜率为零的地方。

![image-20210822143058717](D:\图形学书籍\系列流体文章\gif\image-20210822143058717.png)

虽然arcsim库用了牛顿迭代法，不过直接用牛顿迭代挺慢的，二分法也慢，但把它们结合起来或许会更快一些？safeccd库就是这样做的。selfccd库也用了二分法加牛顿迭代，不过按照那个库的写法，叫牛顿递归更好一些...这两个库的介绍在本篇最后。

总之，二分法配合牛顿迭代法可以用python实现如下

```
def solve_cubic(a,b,c,d):
    # 如果a是0，那么直接解二次方程即可
    if abs(a) < 1e-20:
        return solve_quadratic(b,c,d)
    segment_point_num = 0
    segment_point = np.zeros((4))
    # 计算极值点，把三次方程分成三段
    point_num,point = solve_quadratic(3*a, 2*b, c)
    
    # 没有极值点
    if point_num == 0:
        f0 = d
        f1 = a + b + c + d
        if f0 * f1 < 0:
            segment_point[0] = 0
            segment_point[1] = 1
            segment_point_num = 2
    # 一个极值点
    elif point_num == 1:
        segment_point[0] = 0
        segment_point[1] = point[0]
        segment_point[2] = 1
        segment_point_num = 3
    # 两个极值点
    else:
        segment_point[0] = 0
        segment_point[1] = point[0]
        segment_point[2] = point[1]
        segment_point[3] = 1
        segment_point_num = 4
    
    t_num = 0
    t = np.zeros((3))
    
    for i in range(segment_point_num - 1):
        tLow = segment_point[i]
        tHigh = segment_point[i+1]
        fLow = a*tLow*tLow*tLow + b*tLow*tLow + c*tLow + d
        if abs(fLow) < 1e-10:
            t[i] = tLow
            t_num += 1
            continue
        fHigh = a*tHigh*tHigh*tHigh + b*tHigh*tHigh + c*tHigh + d
        if abs(fHigh) < 1e-10:
            t[i] = tHigh
            t_num += 1
            continue
        
        if tLow > 0:
            temp = tLow
            tLow = tHigh
            tHigh = temp
            
        dx = abs(tHigh - tLow)
        tMid = (tLow + tHigh) / 2
        f = a*tMid*tMid*tMid + b*tMid*tMid + c*tMid + d
        df = 3*a*tMid*tMid + 2*b*tMid + c
        
        for ite in range(100):
            fLow = f - df * (tMid - tLow)
            fHigh = f - df * (tMid - tHigh)
            # 如果解可能隔得近，就用牛顿法
            if fLow * fHigh < 0:
                dx = f / df
                tMid = tMid - dx
                # 如果迭代过头了，就换用二分法
                if tMid >= max(tHigh,tLow) or tMid <= min(tHigh,tLow):
                    dx = (tHigh - tLow) / 2
                    tMid = (tHigh + tLow) / 2
            # 如果解可能隔得远，就用二分法
            else:
                dx = (tHigh - tLow) / 2
                tMid = (tHigh + tLow) / 2
            f = a*tMid*tMid*tMid + b*tMid*tMid + c*tMid + d
            if abs(f) < 1e-10:
                t[i] = tMid
                t_num += 1
                break
            df = 3*a*tMid*tMid + 2*b*tMid + c
            if f < 0:
                tLow = tMid
            else:
                tHigh = tMid
    return t_num,t
```

最后，算出来的一元三次方程的解，仍然可能没有物理意义，所以算出解后仍然要再次验证，在那个时间点，顶点是否碰到了三角形，或者两条边是否相交。

### 代码介绍

最后，这篇文章有个代码仓库，我所写的包括

cubicSolver.py 解一元三次方程

ccdvf.py 三维顶点-面连续碰撞检测

ccdee.py 三维顶点-边连续碰撞检测

除此之外还有一些与连续碰撞检测相关的开源代码包括

github上的

地址：https://github.com/orgs/Continuous-Collision-Detection/repositories

arcsim

地址：http://graphics.berkeley.edu/resources/ARCSim/

简介：开源形变材料模拟引擎

selfccd

地址：https://gamma.cs.unc.edu/SELFCD/

简介：北卡罗来纳大学教堂山分校的开源碰撞检测库中的一个。更多开源库见

https://gamma.web.unc.edu/software/ 。这算得上是除了github上那个外最多的开源碰撞检测库合集了。

safeccd

地址：https://web.cse.ohio-state.edu/~wang.3602/Wang-2014-DCC/

论文：Wang, Huamin. “Defending continuous collision detection against errors.” *ACM Transactions on Graphics (TOG)* 33 (2014): 1 - 10.

lostopos

地址：https://github.com/mtao/LosTopos/blob/master/src/common/cubic_ccd_wrapper.cpp


