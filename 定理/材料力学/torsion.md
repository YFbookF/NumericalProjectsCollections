$$
GI_T \theta' = M_T
$$

Engineering Mechanics 2

equation 5.5 the quantity GI_T is known as torsional rigidity. Given the torque M_T and the torsional rigidity GI_T, we can calculate the angle of twist theta

![image-20211016214511655](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211016214511655.png)
$$
r d\theta = \gamma dx \qquad \tau = G\gamma \qquad \tau = Gr\frac{\partial \theta}{dx} =Gr\theta'
$$
并
$$
M_T = G\theta'\int r^2 dA = G\theta' I_p 
$$
并
$$
\theta_l = \int_0^l \theta' dx \qquad \theta_l = \frac{M_T l}{GI_T} \qquad \tau = \frac{M_T}{I_T}r
$$
https://www.youtube.com/watch?v=1YTKedLQOa0&t=441s

![image-20211017103312475](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211017103312475.png)

gamma = shear strain

![image-20211017103657506](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211017103657506.png)

Basile Audoly, Yves Pomeau - Elasticity and Geometry_ From hair curls to the nonlinear response of shells-Oxford University Press (2010)

Energy of twist

![image-20211027163116540](D:\定理\材料力学\image-20211027163116540.png)

twist Moment\
$$
\bold M(s) = \mu J \tau \bold d_3(s)
$$
traditionally, the component of the internal moment M along the axis d3 of the rod is called the torsional couple, and is denoted H(s)
$$
H(s) = \mu J \tau(s)
$$
孙训方材料力学第五版

薄壁圆筒扭转时，横截面上任一点处的切应力tau 值均相等，其方向与圆周相切，也就是
$$
\int_A \tau dA \times r = T
$$
由于tau为常量，并且对于薄壁圆筒，r可以用平均半径r0 代替，intdA = 2pir0delta是圆筒截面积，壁厚delta，可得，这个公式似乎有问题，你tm是个
$$
\tau = \frac{T}{2A_0 \delta}
$$
再由几何关系，可得薄壁面圆筒表面上的且应变gamma和相距为l的两端面的相对扭转角之间的关系式
$$
l\sin \gamma = r\sin \varphi \qquad l\gamma = r\varphi
$$
相对扭转角与外力偶矩M之间成正比，M在数值上等于扭矩。那么可得
$$
G = \frac{\tau_{xy}}{\gamma _{xy}} = (\frac{F}{A})/(\frac{\Delta x}{h}) = \frac{Fh}{A \Delta x}
$$
![image-20211027173009196](D:\定理\材料力学\image-20211027173009196.png)
$$
\tau = \frac{Tr}{I_p} = \frac{T}{W_p}
$$
Wp是扭转截面系数，T是扭矩，I是极惯性矩。注意下面的
$$
\tau_p = G\gamma = G \rho \frac{d\varphi}{dx}
$$
![+](D:\定理\材料力学\image-20211027193059019.png)

![image-20211027193110747](D:\定理\材料力学\image-20211027193110747.png)
$$
\varphi = \int_0^l d\varphi = \int_0^l \frac{T}{GI_p}dx 
$$
当等直圆杆仅在两端受一对外力偶作用时，则所有横截面上的扭矩T均相同，且等于杆端的外力偶矩M，那么
$$
\varphi = \frac{M_e l}{GI_p} = \frac{Tl}{Gl_p}
$$
上面的GI即为扭转刚度。由于扭转时各横截面上的扭矩可能并不相同，且杆的长度也各不相同，因此使用扭转角对杆长度的变化率来度量
$$
\varphi' = \frac{d \varphi}{dx} = \frac{T}{GI_p}
$$
