基于CUDA的并行程序设计_13572926

![image-20211106180054249](E:\mycode\collection\定理\程序优化\image-20211106180054249.png)

![image-20211106180242093](E:\mycode\collection\定理\程序优化\image-20211106180242093.png)

![image-20211106180255221](E:\mycode\collection\定理\程序优化\image-20211106180255221.png)

一个warp中的线程必然在同一个block中，如果block所含线程数目不是warp大小的整数倍，那么多出的那些thread所在的warp中，会剩余一些inactive的thread，也就是说，即使凑不够warp整数倍的thread，硬件也会为warp凑足，只不过那些thread是inactive状态，需要注意的是，即使这部分thread是inactive的，也会消耗SM资源。**由于warp的大小一般为32，所以block所含的thread的大小一般要设置为32的倍数。**

![image-20211106193107427](E:\mycode\collection\定理\程序优化\image-20211106193107427.png)

- GPU中每个SM都能支持数百个线程并发执行，每个GPU通常有多个SM，当一个核函数的网格被启动的时候，多个block会被同时分配给可用的SM上执行。

**注意:** 当一个blcok被分配给一个SM后，他就只能在这个SM上执行了，不可能重新分配到其他SM上了，多个线程块可以被分配到同一个SM上。

在SM上同一个块内的多个线程进行线程级别并行，而同一线程内，指令利用指令级并行将单个线程处理成流水线。

