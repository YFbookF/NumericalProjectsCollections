https://www.haroldserrano.com/blog/visualizing-the-gjk-collision-algorithm

Simplex 单纯形

0-单纯形就是一个点

1-单纯形就是一条直线

2-单纯形就是一个三角形

3-单纯形就是一个四面体

Supporting Point

对于一个凸多边形和一个方向dir来说，Supporting Point就是在这个方向上最远的点

不会写代码吧？或者想写出来得花很久吧，这就体现先看代码再找代码的好处了。

这样就转化为三次方程在某个区间内有没有根的问题了。



看这个三次方程有没有inflection point，没有的话就可以转换为一个二次方程。



首先如果sign(Y(0)) != sign(Y(1))，那么肯定有根

否则算Y的导数，

http://give.zju.edu.cn/en/memberIntro/tm.html

有的话，先找到这个inflection point，按照左边和右边将这个三次方程分为两段，那么就转化为两个二次方程了。

selfccd 库用的和arcsim差不多

arcsim 用的是普通的三次函数求根

selfccd 算三次函数根的方式是，二分插值+牛顿迭代，并且把牛顿迭代法玩成了牛顿递归法...

deformcd 里的 tri_contact 里有个自称very robust triangle intersection test的算法。

lostopos的代码，也是先算出四个时间分割节点【两两相邻时间节点之间可能碰撞】cubic_ccd_wrapper

https://web.cse.ohio-state.edu/~wang.3602/publications.html

safe ccd

https://web.cse.ohio-state.edu/~wang.3602/Wang-2014-DCC/Wang-2014-DCC_supp.pdf

未知算法，牛顿迭代加二分。过于平缓的就二分，否则就是牛顿迭代。否则如果斜率非常平缓也用牛顿迭代法法的话，容易步子迈的太大。I see I see

```
//**************************************************************************************
//  This cubic solver uses the Newton-Bisection method to find roots within [0, 1].
//	It is based on the implementation in the book "Numerical Recipe in C", but there 
//	are several differences. Please see the paper for more details.
//**************************************************************************************
    void Cubic_Solver(TYPE a, TYPE b, TYPE c, TYPE d, TYPE mu, TYPE root[], int &root_number)
    {
        root_number=0;
        
		//Build the intervals. There are 4 nodes at most.
		TYPE    nodes[4]={0, 0, 0, 0};
        int     node_number=1;
        TYPE    min_max[2];
        int     min_max_number;
        Quadratic_Solver(3*a, 2*b, c, min_max, min_max_number);
        if(min_max_number==2 && min_max[0]>min_max[1])  
		{
			nodes[node_number++]=min_max[1];
			nodes[node_number++]=min_max[0];		
		}
		else
		{
			for(int i=0; i<min_max_number; i++)
				nodes[node_number++]=min_max[i];
		}
        nodes[node_number++]=1;
		        
        //Detect a root in every interval.
        for(int i=0; i<node_number-1; i++)
        {
            TYPE x1=nodes[i];
            TYPE x2=nodes[i+1];
			
			//Obtain lower and upper nodes and their function values.
            TYPE fl=((a*x1+b)*x1+c)*x1+d;
            if(fabs(fl)<mu)     {root[root_number++]=x1; continue;}
            TYPE fh=((a*x2+b)*x2+c)*x2+d;
            if(fabs(fh)<mu)     {root[root_number++]=x2; continue;}
            if(fl>0 && fh>0 || fl<0 && fh<0)    continue;			
			TYPE xl, xh;
            if(fl<0)    {xl=x1; xh=x2;}
            else        {xh=x1; xl=x2;}
			
			//Start with bisection
            TYPE dxold	= fabs(x2-x1);
            TYPE dx		= dxold;
			TYPE rts	= (x1+x2)*0.5;
			TYPE f		= ((a*rts+b)*rts+c)*rts+d;
            TYPE df		= (3*a*rts+2*b)*rts+c;
			
            int j=0;
            for(j=0; j<1024; j++)
            {
				// 比如
                if(((rts-xh)*df-f)*((rts-xl)*df-f)<0 && fabs(2.0*f)<fabs(dxold*df))	//Try Newton first
                {					
                    dxold=dx;
                    dx=f/df;
                    rts=rts-dx;
                    if(rts>=xh && rts>=xl || rts<=xh && rts<=xl) //Switch back to bisection if out of range 	
                    {
                        dxold=dx;
                        dx=0.5*(xh-xl);
                        rts=(xl+xh)*0.5;
                    }
                }
				else //Now do bisection
                {
                    dxold=dx;
                    dx=0.5*(xh-xl);
                    rts=(xl+xh)*0.5;                    
                }
                //Prepare for the next iteration
                f=((a*rts+b)*rts+c)*rts+d;
                if(fabs(f)<mu)     {root[root_number++]=rts; break;}
				df=(3*a*rts+2*b)*rts+c;
                if(f<0) xl=rts;
                else    xh=rts;

            }
            if(j==1024)     printf("ERROR: Fails to converge within 1000 iterations...\n");       
        }
    }
```

