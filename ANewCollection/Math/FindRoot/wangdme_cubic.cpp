//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
///////////////////////////////////////////////////////////////
template <class T> FORCEINLINE
T Solve_Cubic_Equation(const T a, const T b, const T c, const T d, T* roots)
{
    if(a==0)    return  Solve_Quadratic_Equation(b, c, d, roots);
    
    T xn=-b/(3*a);
    T yn=((a*xn+b)*xn+c)*xn+d;
    T yn2=yn*yn;
    T delta2=(b*b-3*a*c)/(9*a*a);
    T h2=4*a*a*delta2*delta2*delta2;
    
    if(yn2>h2)
    {
        T det=sqrt(yn2-h2);
        roots[0]=xn+cbrt((-yn+det)/(2*a))+cbrt((-yn-det)/(2*a));
        return 1;
    }
    if(yn2==h2)
    {
        T delta=cbrt(yn/(2*a));
        roots[0]=xn+delta;
        roots[1]=xn-2*delta;
        return 2;
    }
    else
    {
        T delta=sqrt(delta2);
        T theta=acos(-yn/(2*a*delta*delta2))/3.0;
        roots[0]=xn+2*delta*cos(theta);
        roots[1]=xn+2*delta*cos(theta+2*MY_PI/3.0);
        roots[2]=xn+2*delta*cos(theta+4*MY_PI/3.0);
        return 3;
    }
    return 0;
}
