![image-20211027144711693](D:\定理\材料力学\image-20211027144711693.png)
$$
\varepsilon = \frac{\Delta l}{l_0} = \frac{l - l_0}{l_0} = \frac{(\rho -y)\theta - \rho \theta}{\rho \theta} = -\frac{y}{\rho}
$$
PQ是变形前的neutral axis，P'Q'是变形后的neutral axis。上面这个式子就是说，以中轴为分界线，弯矩为正，那么沿着y轴往上，长度变短。沿着y轴往下，长度变长。

![](D:\定理\材料力学\image-20211027145007854.png)
$$
\sum F_x = 0 \qquad \iint_A dF = \iint _A \sigma dA = 0 \\

$$
注意上面的弯矩，弯矩逆时针为正，所以，三角形支撑的弯矩为负。而且现在为xy平面，如果F的方向为x轴指向，且在y轴上方，那么肯定是负弯矩。

![image-20211027151254132](D:\定理\材料力学\image-20211027151254132.png)
$$
\sum M_z = 0 \qquad -\iint_A ydF - M = -\iint_A y\sigma dA - M = 0
$$
![image-20211027150756118](D:\定理\材料力学\image-20211027150756118.png)

又因为综合以上三式子
$$
\sigma = -E(\frac{y}{\rho}) = E\varepsilon
$$
带入力的积分方程，只有在中轴是centroid的时候成立
$$
\iint ydA = 0
$$
或者
$$
-\iint -yE\frac{y}{\rho}dA - M = 0 \qquad M = \frac{EI}{\rho} \qquad I = \iint_A y^2 dA
$$
说一句，rho是旋转重心到centroid的距离，也就是半径
$$
\frac{d^2y}{dx^2} = \kappa = \frac{M}{EI} = \frac{1}{\rho} \qquad \sigma = -\frac{My}{I}
$$
材料力学孙训方

![image-20211028092454181](D:\定理\材料力学\image-20211028092454181.png)
$$
\varepsilon = \frac{\Delta AB_1}{AB_1} = \frac{BB_1}{AB_1} = \frac{yd\theta}{dx} = \frac{y}{\rho}
$$
表面横截面上任一点处的纵向线应变varepsilon 与该点至中性轴的距离y成正比

![image-20211028092755270](D:\定理\材料力学\image-20211028092755270.png)

最后由静力学公式
$$
F_N = \int_A \sigma dA = \frac{E}{\rho}\int_A ydA = \frac{ES_z}{\rho} = 0 \\ M_y = \int_A z\sigma dA = \frac{E}{\rho}\int_A zydA = \frac{EI_{yz}}{\rho} = 0 \\ M_z = \int y\sigma dA = \frac{E}{\rho}\int y^2 dA = \frac{EI_z}{\rho} = M
$$
最终Sz = 0，也就是z轴必通过横截面的形心。最后可得弯曲刚度EIz，Eiz越大，弯曲变形也就是曲率越小。
$$
\frac{1}{\rho} = \frac{M}{EI_z} \qquad \sigma = \frac{My}{I_z}
$$
材料力学II第一章孙训方

![image-20211028094007671](D:\定理\材料力学\image-20211028094007671.png)

注意两条红线平行且相等
$$
\sigma = \frac{E}{\rho}\eta = \frac{E}{\rho}(y\cos\theta - z\sin\theta)
$$
根据静力学关系可得
$$
F_N = \int_A \sigma dA = 0 \qquad \int_A z\sigma A = M_y \qquad \int_A y\sigma dA = -M_z
$$
![image-20211028094416914](D:\定理\材料力学\image-20211028094416914.png)

