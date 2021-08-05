看起来自动微分最麻烦的是解析字符串？

比如下面的交叉熵损失函数

```
double cross_entropy( const double **a, const double **b )
{
	double loss = 0;
	for ( int i = 0; i < 2; i++ )
	{
		for ( int j = 0; j < 2; j++ )
		{
			loss = loss - (b[i][j] * log( a[i][j] + 0.00001 ) );
		}
	}
	return(loss);
}
```

unroll之后成为

```
double cross_entropy( const double **a, const double **b )
{
	double loss = 0;
	loss	= (loss) - ( (b [0][0]) * (log( (a [0][0]) + (0.00001) ) ) );
	loss	= (loss) - ( (b [0][1]) * (log( (a [0][1]) + (0.00001) ) ) );
	loss	= (loss) - ( (b [1][0]) * (log( (a [1][0]) + (0.00001) ) ) );
	loss	= (loss) - ( (b [1][1]) * (log( (a [1][1]) + (0.00001) ) ) );
	return(loss);
}
```

解析之后如下

```
for(int p = 0; p < num_points ; ++p)
{
ders [i *2+0]+= ((((((((0) - ((( log ((a [0][0] + 0.00001))) * (b [0][0]) +
((1/(( a [0][0] + 0.00001))*0)))))) - ((( log ((a [0][1] + 0.00001))) * (b [
```

并且用到了yacc和lexer，这玩意干的就是解析字符串的活啊。来源

ACORNS: An Easy-To-Use Code Generator for Gradients and
Hessians  

并且cfunction，采取的是字符串形式

```
c_function = "int function_test(double a, double p){ \
    double energy = a*a*a*a*p+1/(p*p) - 1/p * p/a; \
    return 0; \
}"

```

