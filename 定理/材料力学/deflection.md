仅仅使用sigma = E vareps 是不可能算出位移和压力的关系的，因此需要额外的假设

![image-20211016165840029](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211016165840029.png)

这里的w是指点的位移距离

假设一，位移w与z无关，即w = w(x)，也就是截面的高度在弯曲的时候不变化vareps_Z = dw/dz = 0

假设二，平面在经过弯曲后仍然是平面，仅仅是经过一点旋转了。

beam design formulas with shear and moment diagrams

pinned support 只能由向上和向右的力，roller support 只能由向上的力，fixed 只能由向上，向右以及弯矩力

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015150327485.png" alt="image-20211015150327485" style="zoom:50%;" />
$$
V' = q \qquad M' = V \qquad \psi' = \frac{M}{EI}\qquad w' = -\psi \qquad 
$$
那么很容易知道
$$
w'' = \frac{M}{EI}\qquad w''' = -\frac{V}{EI} \qquad w'''' = \frac{q}{EI}
$$
M是弯矩，q是载荷

### 单力

右侧墙壁  单力 

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015135550997.png" alt="image-20211015135550997" style="zoom:50%;" />
$$
EI w'' = F(-x + l) \qquad EIw' = F(-\frac{x^2}{2} + Lx) + C_1\\
EIw = F(-\frac{x^3}{6} + \frac{Lx^2}{2}) + C_1x + C_2
$$
又因为
$$
w'(0) = 0 \qquad w(0) = 0 \rightarrow C_1 = 0 \qquad C_2 = 0
$$
因此slope 和 deflection 分别是
$$
w'(x) = \frac{Fl^2}{2El}(-\frac{x^2}{l^2} + 2\frac{x}{l}) \qquad w(x) = \frac{Fl^3}{6El}(-\frac{x^3}{l^3} + 3\frac{x^2}{l^2})
$$

孙训方材料力学例题5-9

可以用功能原理求

![image-20211028104250816](D:\定理\材料力学\image-20211028104250816.png)

梁任一截面上的弯矩及应变能为
$$
M(x) = Fx \qquad V_e = \int_A \frac{M(x)^2}{2EI}dx = \frac{F^2 l^3}{6EI}
$$
由因为
$$
\frac{1}{2}Fw = V_e = W = \frac{F^2 l^3}{6EI} \qquad w = \frac{Fl^3}{3EI}
$$
左侧墙壁，单力

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015135638438.png" alt="image-20211015135638438" style="zoom:50%;" />

梁任一截面上的弯矩及应变能为
$$
M(x) = -F(L-x) \qquad V_e = \int_A \frac{M(x)^2}{2EI}dx = \frac{F^2}{2EI}(L^3 - L^3 + \frac{L^3}{3})
$$
由因为
$$
\frac{1}{2}Fw = V_e = W =\frac{F^2 L}{6EI} \qquad w = \frac{Fl^3}{3EI}
$$


左侧墙壁，单力

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015135658408.png" alt="image-20211015135658408" style="zoom:50%;" />

右侧墙壁，单力

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015141052374.png" alt="image-20211015141052374" style="zoom:67%;" />

双墙壁单力

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015140725476.png" alt="image-20211015140725476" style="zoom:50%;" />

不知道弯矩为何是FL/8



Engineering Mechanics 2 

![image-20211015162036374](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015162036374.png)

力矩如下
$$
M(x) = \begin{cases} Fbx/l & 0\le x\le a \\ Fa(l-x)/l & a\le x \le l\end{cases}
$$
那么
$$
EIw'' = -\frac{Fbx}{l} \\
EIw' = -F\frac{Fbx^2}{2l} + C_1 \\
EIw = -F\frac{bx^3}{6l} + C_1x + C_2
$$
然后用两端相连

左侧支持，右侧固定

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015141229589.png" alt="image-20211015141229589" style="zoom:67%;" />

两力三支持

![image-20211015141724054](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015141724054.png)

==========A THREE-DIMENSIONAL FINITE-STRAIN ROD MODEL  

![image-20211026144059697](D:\定理\材料力学\image-20211026144059697.png)

### 均匀分布力

首先记住
$$
EIw'''' = q = q_0 \\
EIw''' = -V = q_0x + C_1 \\
EIw'' = -M = \frac{1}{2}q_0 x^2  +C_1x + C_2\\
EIw' = \frac{1}{6}q_0 x^3 + \frac{1}{2}C_1 x^2 + C_2x + C_3\\
EIw = \frac{1}{24}q_0 x^4 + \frac{1}{6}C_1 x^3 + \frac{1}{2}C_2 x^2 + C_3 x + C_4
$$
左侧墙壁，均匀分布力

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015135750887.png" alt="image-20211015135750887" style="zoom:50%;" />

Engineering Mechanics 2
$$
w'(0) = 0 = C_3 \\
w(0) = 0 = C_4 \\
V(l) = 0  \rightarrow q_0 l + C_1 = 0 \rightarrow C_1 = -q_0l \\
M(l) = 0 \rightarrow \frac{1}{2}q_0l^2 + C_1l + C_2 = 0 \rightarrow C_2 = \frac{1}{2}q_0l^2
$$
所以deflection
$$
w(x) = \frac{q_0l^4}{24EI}((\frac{x}{l})^4 - 4(\frac{x}{l})^3 + 6(\frac{x}{l})^2)
$$
右侧墙布，均匀分布力https://eng.libretexts.org/Bookshelves/Mechanical_Engineering/Introduction_to_Aerospace_Structures_and_Materials_(Alderliesten)/02%3A_Analysis_of_Statically_Determinate_Structures/04%3A_Internal_Forces_in_Beams_and_Frames/4.04%3A_Relation_Among_Distributed_Load_Shearing_Force_and_Bending_Moment

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015135942982.png" alt="image-20211015135942982" style="zoom:50%;" />

双墙壁

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015140753197.png" alt="image-20211015140753197" style="zoom:50%;" />

![image-20211018212922727](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211018212922727.png)

双支持力，看Engineering Mechanics 2 p132很详细

![image-20211015153135027](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015153135027.png)

$$
w(x) = \frac{q_0 l^4}{24EI}[(\frac{x}{l})^4 - 2(\frac{x}{l})^3 + (\frac{x}{l})] \\
w(x)' = \frac{q_0}{24EI}[4x^3 - 6x^2l + l^3]
$$
这就是为什么p142会如此写
$$
EIw'(0) = EIw'_A = \frac{q_0 l^3}{24} \qquad EIw'(l) = EIw'_B = -\frac{q_0 l^3}{24}
$$
左侧支持力，右侧固定，左侧没有bending moment

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015141121905.png" alt="image-20211015141121905" style="zoom: 67%;" />

不同侧支持力

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015141344429.png" alt="image-20211015141344429" style="zoom:67%;" />

仅突出部分有力

![image-20211015141607835](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015141607835.png)

### 非均匀分布力

左侧墙壁，非均匀分布力

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015135814414.png" alt="image-20211015135814414" style="zoom:67%;" />

首先
$$
EIw'''' = w_1(1- \frac{x}{L})\\
EI w''' = -V = w_1(x - \frac{x^2}{2L}) + C_1\\
EI w'' = -M = w_1(\frac{x^2}{2} - \frac{x^3}{6L}) + C_1x + C_2\\
EI w' = w_1(\frac{x^3}{6} - \frac{x^4}{24L}) +\frac{x^2}{2}C_1 + C_2x + C_3\\
EI w = w_1(\frac{x^4}{24} - \frac{x^5}{120L}) + \frac{x^3}{6}C_1 + \frac{x^2}{2}C_2 + C_3x + C_4
$$
又
$$
w'(0) = 0\qquad w(0) = 0 \qquad C_3 = C_4 = 0 \qquad M(L) = 0 \qquad V(L) = 0
$$
那么
$$
C_1 = -\frac{w_1L}{2} \qquad C_2 = \frac{w_1 L^2}{6}
$$
复杂一点的

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015142233701.png" alt="image-20211015142233701" style="zoom: 50%;" />

双墙壁

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015155942997.png" alt="image-20211015155942997" style="zoom:50%;" />



### 转矩

单转矩在中间，剪力图无变化，弯矩图因为顺时针下降

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015104318129.png" alt="image-20211015104318129" style="zoom: 50%;" />

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211018213024080.png" alt="image-20211018213024080" style="zoom:67%;" />

![image-20211025191144057](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211025191144057.png)

2

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211018213048887.png" alt="image-20211018213048887" style="zoom:67%;" />

3

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211018213104108.png" alt="image-20211018213104108" style="zoom:67%;" />

5

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211018213124779.png" alt="image-20211018213124779" style="zoom:67%;" />

单转矩在左侧，剪力图无变化，弯矩图因为逆时针上升

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015140525852.png" alt="image-20211015140525852" style="zoom:50%;" />

双转矩在左右两侧

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015140611521.png" alt="image-20211015140611521" style="zoom:50%;" />

单转矩在中间

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015140631931.png" alt="image-20211015140631931" style="zoom:50%;" />

Engineering Mechanics 2

<img src="C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015155540315.png" alt="image-20211015155540315" style="zoom:50%;" />

此时弯矩是个变量
$$
M(x) = M_0\frac{x}{l}
$$
那么
$$
EIw'' = -\frac{M_0}{l}x \\
EIw' = -\frac{M_0}{2l}x^2 + C_1\\
EIw = -\frac{M_0}{6l}x^3 + C_1x + C_2
$$
结果没写

![image-20211018212804356](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211018212804356.png)

![image-20211018212935592](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211018212935592.png)

A THREE-DIMENSIONAL FINITE-STRAIN ROD MODEL  

![image-20211026143904386](D:\定理\材料力学\image-20211026143904386.png)
