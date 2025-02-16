/*********************************************************************
 * oscillator.h
 * Authored by Kris Hauser 2002-2003
 *
 * Header for the Oscillator class, an efficient 
 * encapsulator of the solution to a second-order
 * differential equation.
 *
 * Copyright 2003, Regents of the University of California 
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *********************************************************************/


//8/12/02: Kris Hauser, SCEA
//Modified to allow hard deformation limits via the Crop...() functions

#ifndef OSCILLATOR_H
#define OSCILLATOR_H

#ifdef MATH_DOUBLE
typedef double float_t;
#else
typedef float float_t;
#endif


class Oscillator
{
public:
	Oscillator();

	/*******************************INITIALIZERS:******************************/
	//solves Kx + M(dx/dt) + C(d2x/dt2) = 0  =>  e ^(-c +- sqrt(c^2 - 4km) / 2)
	void Setup_KMC(float_t K, float_t M, float_t C);

	// =>  e ^(-R +- (R - 1)^(1/2) * wn)	where wn = sqrt(k/m), R = c / c(critical) = c / sqrt(4km)
	void Setup_DampingRatioNaturalFrequency(float_t R, float_t wn);

	// => e^w1 + e^w2
	void Setup_Omega1Omega2(float_t w1_real, float_t w2_real, float_t w1_imag = 0.0, float_t w2_imag = 0.0);

	// => c = a1*k + a2*m
	void Setup_KMa1a2(float_t K, float_t M, float_t alpha1, float_t alpha2);

	void Setup_dt(float_t dt);  //Sets the default time step.  Call this after one of the previous initializers.


	/*******************************METHODS:******************************/
	void Inc_t();				//Increments by the pre-set dt.  Using this is the most efficient way to update.
	void Inc_t(float_t dt);		//Increments by an arbitrary dt.  (Does not modify the old dt)

	inline float_t Get_z() const		//Returns the current state of z.  Using this is the most efficient way to get the z value.
		{ return p1 + p2; }
	float_t Get_z(float_t t) const;		//Solves for z at time t (relative to the current time, i.e. current time = 0).
	float_t Get_dz() const;				//Returns the current state of dz.
	float_t Get_ddz() const;                        //Returns the current state of ddz

	void Set_z_dz (float_t z, float_t dz);		//Sets the state of the oscillator to have the provided z and dz values.
	void Inc_z_dz (float_t z, float_t dz);		//Increments the current state by the provided z and dz values.


	float_t GetDecay1() const;		//Returns the decay of the first phasor
	float_t GetDecay2() const;		//Returns the decay of the second phasor
	float_t GetFreq() const;		//Returns the frequency, in Hz

	inline bool UnderDamped() const
		{ return wd > 0.0; }
	inline bool OverDamped() const
		{ return wd < 0.0; }
	inline bool CriticallyDamped() const
		{ return wd == 0.0; }

	float_t Get_z_df() const;		//Returns the change of z with respect to a change in dz
	float_t Get_dz_df() const;		//Returns the change of dz with respect to a change in dz

	//Kris' SCEA additions
	void Crop_Set(float_t& z, float_t& dz, float_t limit) const;
	void Crop_Inc(float_t& z, float_t& dz, float_t limit) const;

private:
	void set_ab(float_t a, float_t b);
	void get_deltas(float_t dt, float_t& s, float_t& t, float_t& u, float_t& v) const;
	inline void update_state(float_t s, float_t t, float_t u, float_t v)
	{
		float_t oldreal = p1;
		float_t oldimag = q1;
		p1 = oldreal*s - oldimag*t;
		q1 = oldreal*t + oldimag*s;

		oldreal = p2;
		oldimag = q2;
		p2 = oldreal*u - oldimag*v;
		q2 = oldreal*v + oldimag*u;
	}

	float_t p1, q1;			//the current state (phasor 1)
	float_t dp1, dq1;		//the change in state for each dt  (phasor 1)
	float_t p2, q2;			//the current state (phasor 2)
	float_t dp2, dq2;		//the change in state for each dt  (phasor 2)

	float_t d1, d2;			//the decay of the phasors. = (-c +- sqrt(c^2 - 4km) / 2) if real, = (-c/2) if imag
	float_t f1, f2;			//the frequency of the phasors.  = 0 if real, = sqrt(c^2 - 4km)/2 if imag
	float_t wd;				//angular freq = sqrt(c^2 - 4km)
};

#endif
