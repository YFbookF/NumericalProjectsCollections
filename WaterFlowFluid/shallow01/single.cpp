// minimal shallow water solver by Nils Thuerey
// see http://www.ntoken.com and http://www.matthiasmueller.info/realtimephysics/ for further info
// this code is released under the GPL, see LICENSE.txt for details
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

// global parameters
const int SIZE = 100;

const int IMAGESTEPS = 2;

const int END_TIME = 25;

// single / double precision?
typedef float Real;

// use the turbulence model?
const bool SMAGORINSKY = false;

const int FLUID = 0, NOSLIP = 1, VELOCITY = 2;
const Real LDC_VELOCITY = 0.10;

// constants
const Real gravity = -10.;  // normal acceleration a_n
const Real dt = 10. / (Real)SIZE;       // time step size
const Real domainSize = 50.; // size of the domain
const Real dx = domainSize / (Real)SIZE;
const Real dxInv = 1. / dx;   // save some division later on

// impose global velocity
const Real GLOBAL_U = 0.;
const Real GLOBAL_V = 0.;

// main arrays velocity, height, and temporary storage
std::vector<Real> vel_x, vel_y, temp, eta;

void writeImage(int s);
void addRandomDrop();

Real interpolate(std::vector<Real>& array, Real x, Real y) {
	const int X = (int)x;
	const int Y = (int)y;
	const Real s1 = x - X;
	const Real s0 = 1.f - s1;
	const Real t1 = y - Y;
	const Real t0 = 1.f - t1;
	return  s0 * (t0 * array[X + SIZE * Y] + t1 * array[X + SIZE * (Y + 1)]) +
		s1 * (t0 * array[(X + 1) + SIZE * Y] + t1 * array[(X + 1) + SIZE * (Y + 1)]);
}

void advectArray(std::vector<Real>& array, int velocityType) {
	for (int i = 1; i < SIZE - 1; i++)
		for (int j = 1; j < SIZE - 1; j++) {
			const int index = i + j * SIZE;

			// distinguish different cases to interpolate the velocity
			Real u = GLOBAL_U, v = GLOBAL_V;
			switch (velocityType) {
			case 0: // heights
				u += (vel_x[index] + vel_x[index + 1]) * 0.5;
				v += (vel_y[index] + vel_y[index + SIZE]) * 0.5;
				break;
			case 1: // x velocity
				u += vel_x[index];
				v += (vel_y[index] + vel_y[index + 1] + vel_y[index + SIZE] + vel_y[index + SIZE + 1]) * 0.25;
				break;
			case 2: // y velocity
				u += (vel_x[index] + vel_x[index + 1] + vel_x[index + SIZE] + vel_x[index + SIZE + 1]) * 0.25;
				v += vel_y[index];
				break;
			default: // invalid
				exit(1);
			}

			// backtrace position
			Real srcpi = (Real)i - u * dt * dxInv;
			Real srcpj = (Real)j - v * dt * dxInv;

			// clamp range of accesses
			if (srcpi < 0.) srcpi = 0.;
			if (srcpj < 0.) srcpj = 0.;
			if (srcpi > SIZE - 1.) srcpi = SIZE - 1.;
			if (srcpj > SIZE - 1.) srcpj = SIZE - 1.;

			// interpolate source value
			temp[index] = interpolate(array, srcpi, srcpj);
		}

	// copy back...
	for (int i = 1; i < SIZE - 1; i++)
		for (int j = 1; j < SIZE - 1; j++) {
			const int index = i + j * SIZE;
			array[index] = temp[index];
		}
}


void updateHeight() {
	// update heights
	for (int i = 1; i < SIZE - 1; i++)
		for (int j = 1; j < SIZE - 1; j++) {
			const int index = i + j * SIZE;
			Real dh = -0.5 * eta[index] * dxInv * (
				(vel_x[index + 1] - vel_x[index]) +
				(vel_y[index + SIZE] - vel_y[index]));

			eta[index] += dh * dt;
		}
}

void updateVelocities() {
	// accelerate x velocity
	for (int i = 2; i < SIZE - 1; i++)
		for (int j = 1; j < SIZE - 1; j++) {
			const int index = i + j * SIZE;

			vel_x[index] += gravity * dt * dxInv * (
				(eta[index] - eta[index - 1]));
		}

	// accelerate y velocity
	for (int i = 1; i < SIZE - 1; i++)
		for (int j = 2; j < SIZE - 1; j++) {
			const int index = i + j * SIZE;

			vel_y[index] += gravity * dt * dxInv * (
				(eta[index] - eta[index - SIZE]));
		}
}

void setBoundary() {
	// set reflecting boundaries
	for (int i = 0; i < SIZE; i++) {
		const int index1 = i + 0 * SIZE;
		const int index2 = i + (SIZE - 1) * SIZE;
		eta[index1] = eta[index1 + SIZE];
		eta[index2] = eta[index2 - SIZE];
	}
	for (int j = 0; j < SIZE; j++) {
		const int index1 = 0 + j * SIZE;
		const int index2 = (SIZE - 1) + j * SIZE;
		eta[index1] = eta[index1 + 1];
		eta[index2] = eta[index2 - 1];
	}
}


// run the solver...
int main(int argc, char* argv[])
{
	printf("Init\n");

	// allocate memory
	vel_x.resize(SIZE * SIZE);
	vel_y.resize(SIZE * SIZE);
	temp.resize(SIZE * SIZE);
	eta.resize(SIZE * SIZE);

	// initialize grid
	for (int i = 0; i < SIZE; i++)
		for (int j = 0; j < SIZE; j++) {
			const int index = i + j * SIZE;
			vel_x[index] =
				vel_y[index] =
				temp[index] = 0.;

			// default water height 1
			eta[index] = 1.;
		}

	// init an off-center initial wave
	for (int i = 3 * SIZE / 6; i < 4 * SIZE / 6; i++)
		for (int j = 3 * SIZE / 6; j < 4 * SIZE / 6; j++) {
			const int index = i + j * SIZE;
			//eta[index] = 1.1; 
		}

	printf("Starting simulation...\n");

	Real simulationTime = 0.;
	int  simulationStep = 0;

	while (simulationTime < END_TIME)
	{
		printf("Step %d ", simulationStep);

		advectArray(eta, 0);
		advectArray(vel_x, 1);
		advectArray(vel_y, 2);

		updateHeight();
		updateVelocities();

		setBoundary();
		addRandomDrop();

		simulationTime += dt;
		simulationStep++;

		// should we write an image for this step?
		if ((simulationStep % IMAGESTEPS) == 0) {
			writeImage(simulationStep);
		}
		printf("\n");
	}

	printf("SWS-simulation done!\n");
	return 0;
}


inline Real getRandom(Real min = 0., Real max = 1.) {
	return min + ((rand() / (RAND_MAX + 1.0)) * (max - min));
}

// add a random elevation to create some waves
void addRandomDrop() {
	if (getRandom() <= 0.25 * dt) {
		int px = getRandom() * (Real)SIZE;
		int py = getRandom() * (Real)SIZE;
		int s = getRandom() * (Real)SIZE * 0.05;
		Real h = getRandom(0.01, 0.07);
		if (s < 1) s = 1;

		for (int i = px - s; i < px + s; i++)
			for (int j = py - s; j < py + s; j++) {
				const int index = i + j * SIZE;
				if (i<0 || j<0 || i>(SIZE - 1) || j>(SIZE - 1))
					continue;
				eta[index] += h;
				if (eta[index] > 1.1) eta[index] = 1.1;
			}
	}
}

// helper, clamp a in range b<= a <= c, write to a
template < class T > inline T CLAMP(T a, T b, T c) {
	if (a < b) a = b;
	else if (a > c) a = c;
	return a;
}
template < class T > inline T CLAMP255(T a) {
	return CLAMP(a, 0, 255);
}

// helper function to write out the velocity field
// or densities to an image
#define OUTNAME "outsws"
void writeImage(int s) {
	// pressure or velocities?
	const bool showVelocity = true;

	// open output file 
	FILE* outfile;
	char outfilename[256];
	sprintf(outfilename, OUTNAME "_%04d.ppm", s);
	if ((outfile = fopen(outfilename, "w")) == NULL) {
		printf("Unable to open file '%s'...\n", outfilename);
		return;
	}

	// write ppm header
	fprintf(outfile, "P3\n%d %d\n%d\n", SIZE, SIZE, 255);

	const Real hRange = 0.025;
	const Real vRange = 0.015;
	const Real minH = 1. - hRange;  // manually set height range for picture
	const Real maxH = 1. + 2. * hRange;
	const Real minVel = 0. - vRange;  // manually set velocity range for picture
	const Real maxVel = 0. + vRange;

	for (int j = 0; j < SIZE; j++) {
		for (int i = 0; i < SIZE; i++) {
			Real h = eta[i + j * SIZE];
			int grey = (int)((h - minH) / (maxH - minH) * 255.0);
			if (grey < 0) grey = 0;
			if (grey > 255) grey = 255;
			int r = grey;
			int g = grey;
			int b = grey;

			if (showVelocity) {
				Real u_x = vel_x[i + j * SIZE];
				Real u_y = vel_y[i + j * SIZE];

				r += (int)((u_x - minVel) / (maxVel - minVel) * 100.0);
				g += (int)((u_y - minVel) / (maxVel - minVel) * 100.0);
			}

			fprintf(outfile, "%4d %4d %4d ", CLAMP255(r), CLAMP255(g), CLAMP255(b));
		}
		fprintf(outfile, "\n");
	}

	fclose(outfile);
	// image written...		
	printf(" wrote '%s' ", outfilename);

	// try to convert, assumes you have ImageMagick
	char convert[256];
	sprintf(convert, "convert " OUTNAME "_%04d.ppm " OUTNAME "_%04d.png ", s, s);
	system(convert);
}



