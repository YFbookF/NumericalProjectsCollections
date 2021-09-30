![image-20210929224126516](D:\图形学书籍\系列流体文章\gif\image-20210929224126516.png)

![image-20210929224150643](D:\图形学书籍\系列流体文章\gif\image-20210929224150643.png)

```
void swap(int *a,int *b)
{
	int temp;
	temp = *a;
	*a = *b;
	*b = temp;
}
swap(&a,&b);
```

```
p[i] 是 *(p + i)的简便写法
```

![image-20210929225335301](D:\图形学书籍\系列流体文章\gif\image-20210929225335301.png)

![image-20210929230710327](D:\图形学书籍\系列流体文章\gif\image-20210929230710327.png)

函数指针

```
#include <stdio.h>
#include <string.h>
char * fun(char * p1, char * p2)
{
    int i= 0;
    i = strcmp(p1,p2);
    if (0 == i)
    {
        return p1;
    }
    else
    {
        return p2;
    }
}

int main()
{
    char * (*pf)(char * p1,char * p2);
    pf = &fun;
    (*pf) ("aa","bb");
    return 0;
}

作者：linux亦有归途
链接：https://zhuanlan.zhihu.com/p/317301422
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

![image-20210929233407363](D:\图形学书籍\系列流体文章\gif\image-20210929233407363.png)

传统指针的弊端

- 没有说明指向的是单个物体还是数组
- 在声明的时候，没有指明指针是否拥有那个物体。
- 如果你希望销毁指针指向的物体，其实没有办法做到。应该使用delete还是另一套机制。
- 如果使用delete，那么对于单个物体应该使用delete，对于多个物体应该使用delete[]。如果使用了错误的形式，那么结果则是未定义。
- 就算你知道指针指向什么物体，也知道如何销毁它，你也不能保证delete函数真的只执行了一次。少一个则代表资源泄露，多一个则代表未定义行为。
- 没办法知道指针何时悬空，即不知道指针指向的物体的内存不再拥有物体

std::unique_ptr embodies exclusive ownership semantics. A non-null std::
unique_ptr always owns what it points to. Moving a std::unique_ptr transfers
ownership from the source pointer to the destination pointer. (The source pointer is set to null.) Copying a std::unique_ptr isn’t allowed, because if you could copy a std::unique_ptr, you’d end up with two std::unique_ptrs to the same resource, each thinking it owned (and should therefore destroy) that resource.
std::unique_ptr is thus a move-only type. Upon destruction, a non-null
std::unique_ptr destroys its resource. By default, resource destruction is accom‐
plished by applying delete to the raw pointer inside the std::unique_ptr.  

• std::unique_ptr is a small, fast, move-only smart pointer for managing
resources with exclusive-ownership semantics.
• By default, resource destruction takes place via delete, but custom deleters
can be specified. Stateful deleters and function pointers as deleters increase the
size of std::unique_ptr objects.
• Converting a std::unique_ptr to a std::shared_ptr is easy.  

If sp1 and sp2 are std::shared_ptrs to
different objects, the assignment “sp1 = sp2;” modifies sp1 such that it points to the
object pointed to by sp2. The net effect of the assignment is that the reference count
for the object originally pointed to by sp1 is decremented, while that for the object
pointed to by sp2 is incremented.  

std::shared_ptrs是野指针的两倍，而reference count的内存必须动态收集。因为指针知道自己存的物体，而存的物体不知道指针的情况。使用std::make_shared可以避免这个情况。

必须使用原子操作，因为有多线程的情况出现。

使用std::shared_ptr通常需要增加reference count。不通常的情况则是move Construction

![image-20210930091729581](D:\图形学书籍\系列流体文章\gif\image-20210930091729581.png)

```
std::shared_ptr<Widget> spw1(new Widget loggingDel);
std::shared_ptr<Widget> spw2(spw1);
```

• std::shared_ptrs offer convenience approaching that of garbage collection
for the shared lifetime management of arbitrary resources.
• Compared to std::unique_ptr, std::shared_ptr objects are typically
twice as big, incur overhead for control blocks, and require atomic reference
count manipulations.
• Default resource destruction is via delete, but custom deleters are supported.
The type of the deleter has no effect on the type of the std::shared_ptr.
• Avoid creating std::shared_ptrs from variables of raw pointer type.  

https://zhuanlan.zhihu.com/p/365414133#:~:text=weak_ptr,red_ptr%E3%80%82

http://avdancedu.com/a39d51f9/

It’s useful to approach std::move and std::forward in terms of what they don’t do.
std::move doesn’t move anything. std::forward doesn’t forward anything. At run‐
time, neither does anything at all. They generate no executable code. Not a single
byte.  

• std::move performs an unconditional cast to an rvalue. In and
doesn’t move anything.
• std::forward casts its argument to an rvalue only if that argume
to an rvalue.
• Neither std::move nor std::forward do anything at runtime  