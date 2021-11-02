https://mechanicalc.com/reference/beam-analysis

moment curvature equation
$$
\frac{1}{R} = \frac{M}{EI}
$$
而且
$$
\frac{1}{R} = \kappa = \frac{d^2 y/dx^2}{(1+(dy/dx)^2)^{3/2}} = \frac{M}{EI} \approx \frac{d^2 y}{dx^2}
$$
也就是
$$
\frac{dy}{dx} = \frac{1}{EI}\int M(x)dx = \frac{1}{EI}\int (-Px)dx =\frac{1}{EI}(-\frac{P}{2}x^2)
$$
P 就是pressure or force

![image-20211015102911175](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015102911175.png)

那么
$$
y = \frac{1}{EI}(-\frac{P}{6}x^3 + C_1x + C_2)
$$
并且，根据边界条件可得，注意边界条件，上面那个网站边界条件有问题
$$
y = \frac{P}{6EI}(-x^3 + 3L^2x - 2L^3)
$$
bending moment 越大，curvature 越大，radius 越小

![image-20211015103155561](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015103155561.png)

那么 bending moment 如下

![image-20211015103340045](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015103340045.png)

![image-20211015103840314](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015103840314.png)

![image-20211015104318129](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211015104318129.png)

那么之前那个也就是
$$
y_1 = \frac{1}{EI}(\frac{P}{12}x^3 + C_1x + C_2) \qquad (0 \le x \le L/2)
$$
以及
$$
y_2 = \frac{1}{EI}(-\frac{P}{12}x^3 + \frac{PL}{4}x^2 + C_3x + C_4)
$$
