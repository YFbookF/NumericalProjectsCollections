A GPU consists of a large set of streaming multiprocessors (SMs). Since
each SM is essentially a multicore machine in its own right, you might say
the GPU is a multi-multicore machine. Though the SMs run independently,
they share the same GPU global memory. On the other hand, unlike ordinary multicore systems, there are only very limited (and slow) facilities for
barrier synchronization and the like  

GPU包含了大量的流多处理器streaming multiprocessors。每个SM自己都相当于一个多核机器。尽管每个SM独立运行，它们都共享相同的GPU全局内存。并且只有非常有限的慢速的设施用于屏障同步。

每个SM都包含了一堆SP，也就是独立核心。这个核心用于运行线程。不同SM中的线程不能同步。

当编写CUDA程序的时候，将线程组织起来的最后都装到了blocks中。要点如下

- 硬件将整个block放在一个SM上，但是一个SM上可以有很多个block
- 同一个block之间的线程是可以barrier synchronization
- 同一个block之中的线程可以访问由开发者管理的缓存，叫做shared memory
- 硬件将每个block划分为warps
- Thread scheduling is handled on a warp basis. When some cores
  become free, this will occur with a set of 32 of them. The hardware
  then finds a new warp of threads to run on these 32 cores.  
- 