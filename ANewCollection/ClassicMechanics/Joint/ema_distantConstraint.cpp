https://github.com/ema0j/distance-constraint/blob/master/pj_step0/constraint.cpp
#include "constraint.h"

#define iter 3
#define time 0.001
#define bias 0.01

void setJacobian(vec4* jacobian, vec4 x1, vec4 x2)
{
	jacobian[0] = vec4(0.0);
	jacobian[1] = x2 - x1;
	//for(int i = 0; i<2; i++)
	//{
	//	jacobian[i] = 0.0;
	//	jacobian[i+3] = x2[i] - x1[i];
	//}

}

float calcA(vec4* jacobian)
{
	float A = 0.0;
	for(int i=0; i<2; i++)
	{
		A = A + jacobian[i].x * 0.5 * jacobian[i].x
			  + jacobian[i].y * 0.5 * jacobian[i].y
			  + jacobian[i].z * 0.5 * jacobian[i].z;
	}

	return A;
}

float calcB(vec4* jacobian, vec4* v)
{
	float B = 0.0;
	for(int i=0; i<2; i++)
	{
		B = B + jacobian[i].x * v[i].x
			  + jacobian[i].y * v[i].y
			  + jacobian[i].z * v[i].z;
	}

	return -B/time;
}

float calcC(vec4 x1, vec4 x2, float L)
{
	float pp, C = 0.0;
	vec4 p;

	p = x1 - x2;
	pp = p.x*p.x + p.y*p.y + p.z*p.z;
	C = sqrt(pp) - L;

	return -(bias*C)/(time*time);
}

void calcV2(vec4* v2, vec4* jacobian, vec4* v1, float lambda)
{
	for(int i=0; i<2; i++)
		v2[i] = jacobian[i] * lambda * 0.5 * time + v1[i];
}

vec4 calcX2(vec4 x, vec4 v2)
{
	vec4 x2 = x + v2 * time;

	return x2;
}

void applyConstraint(vec4 dst, vec4 vd, vec4 pos, vec4 vp,
					vec4* post_dst, vec4* vd2, vec4* post_pos, vec4* vp2)
{
	int i, j;
	float L = 0.0;
	vec4 jacobian[2];

	float m1 = 2, m2 = 2;
	vec4 v[2] = { vd, vp };
	vec4 p1 = dst, p2, q1 = pos, q2;
	vec4 v2[2];

	//float origin[3] = {0.0, 0.0, 0.0};
	//float x[3] = {10.0, 10.0, 0.0};
	//float v[6] = {0.0, 0.0, 0.0, 0.1, 0.1, 0.1};
	//float origin2[3], x2[3], v2[6];

	float A, B, C, lambda, delta_lambda;
	lambda = 0.1;

	//for(i=0; i<40; i++)
	//{
		setJacobian(jacobian, p1, q1);

		A = calcA(jacobian);
		B = calcB(jacobian, v);
		C = calcC(p1, q1, L);
		B = B + C;

		for(j=0; j<iter; j++)
		{
			delta_lambda = (float)(B - A*lambda)/A;
			lambda = lambda + delta_lambda;
		}

		calcV2(v2, jacobian, v, lambda);
		p2 = calcX2(p1, v2[0]); 
		q2 = calcX2(q1, v2[1]);

		*post_dst = p2;
		*vd2 = v2[0];
		*post_pos = q2;
		*vp2 = v2[1];

	//cout << "vp " << *vd2 << endl;
	//cout << "vt " << *vp2 << endl;
	//cout << "point " << *post_dst << endl;
	//cout << "tri_point " << *post_pos << endl;
	//cout << "**************** " << cnt++ << " ***************" << endl;
	//if( cnt == 2 ) exit(1);
