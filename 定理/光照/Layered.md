=======================A Comprehensive Framework for Rendering Layered Materials  

光照到表面上的时候，有R的概率反射，有T的概率进入材料，但如果是多层材料，接下来的一层也照样有R的概率反射，有T的几率继续进入材料

![image-20211116150337611](E:\mycode\collection\定理\光照\image-20211116150337611.png)
$$
\tilde R = R + TRT + ... = R + \frac{RT^2}{1 - R^2} \qquad \tilde T = TT + TR^2T + ... = \frac{T^2}{1 - R^2}
$$
