// minimal LBM D3Q19 solver by Nils Thuerey
// see http://www.ntoken.com and http://www.matthiasmueller.info/realtimephysics/ for further info
// this code is released under the GPL, see LICENSE.txt for details
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// global parameters
const int SIZE = 32;

// for a more turbulent simulation,
// try: SMAGORINSKY=true, omega=2, LDC_VELOCITY=0.166


// single / double precision?
typedef float Real;

// use the turbulence model?
const bool SMAGORINSKY = false;

// different types of cells
const int FLUID = 0, NOSLIP = 1, VELOCITY = 2;
// velocity for the lid-driven-cavity setup
const Real LDC_VELOCITY = 0.10;

// Declare some constants and globals ...
const Real   omega = 1.90;     // viscosity of the fluid, 0..2
const int    STEPS = 1000;     // max no. of steps to simulate
const int    IMAGESTEPS = 20;  // write image every ? steps
int	         current = 0, other = 1; // which grid is the current and the old one?

// main arrays (note - for simplicity these are allocated here, for larger 
//              simulations, better use malloc and a manual offset calculation)
Real   cells[2][SIZE][SIZE][SIZE][19]; // the two grids of LBM cells with 19 distribution functions
char   flags[SIZE][SIZE][SIZE];        // flags for each cell either FLUID, NOSLIP or VELOCITY

// the weight for the equilibrium distribution
Real w[19] = { (1. / 3.),
	(1. / 18.),(1. / 18.),(1. / 18.),(1. / 18.),(1. / 18.),(1. / 18.),
	(1. / 36.),(1. / 36.),(1. / 36.),(1. / 36.),(1. / 36.),(1. / 36.),
	(1. / 36.),(1. / 36.),(1. / 36.),(1. / 36.),(1. / 36.),(1. / 36.), };

// convenience variables for handling the distribution functions in a loop
// the 19 lattice vectors
int e_x[19] = { 0, 0,0,  1,-1, 0,0,  1,-1,1,-1, 0,0,0,0,   1,1,-1,-1 };
int e_y[19] = { 0, 1,-1, 0,0,  0,0,  1,1,-1,-1, 1,1,-1,-1, 0,0,0,0 };
int e_z[19] = { 0, 0,0,  0,0,  1,-1, 0,0,0,0,   1,-1,1,-1, 1,-1,1,-1 };
// the index of the inverse for each lattice vector
int dfInv[19] = { 0 ,  2 , 1 , 4 , 3 , 6 , 5 , 10 , 9 , 8 , 7 , 14 , 13 , 12 , 11 , 18 , 17 , 16 , 15 };

// helper functions for turbulence model
#include "smagorinksy.h"

void writeImage(int s);


// run the solver...
int main(int argc, char* argv[])
{
	printf("Init\n");
	if (SMAGORINSKY) initSmagoArrays();

	// initialize grid
	const int BORDER = 3;
	for (int i = 0; i < SIZE; i++)
		for (int j = 0; j < SIZE; j++)
			for (int k = 0; k < SIZE; k++) {
				for (int l = 0; l < 19; l++) {
					cells[0][i][j][k][l] = cells[1][i][j][k][l] = w[l];
				}
				if ((i == 0) || (i == SIZE - 1) || (j == 0) || (j == SIZE - 1) || (k == 0) || (k == SIZE - 1)) {
					flags[i][j][k] = NOSLIP;
				}
				else if (j == SIZE - 2 &&
					i > BORDER && i<SIZE - (1 + BORDER) &&
					k>BORDER && k < SIZE - (1 + BORDER)) {
					// lid driven cavity - fixed velocity at top layer
					// BORDER keeps velocities from being forced directly into the wall
					flags[i][j][k] = VELOCITY;
				}
				else {
					// everything else is fluid
					flags[i][j][k] = FLUID;
				}
			}

	printf("Starting simulation...\n");

	for (int s = 0; s < STEPS; s++)
	{
		printf("Step %d\n", s);

		// Stream from the other to the current grid...
#		pragma omp parallel
		{

#		pragma omp for  schedule(static)
			for (int i = 1; i < SIZE - 1; i++)
				for (int j = 1; j < SIZE - 1; j++)
					for (int k = 1; k < SIZE - 1; k++)
						for (int l = 0; l < 19; l++) {

							int inv = dfInv[l];
							if (flags[i + e_x[inv]][j + e_y[inv]][k + e_z[inv]] == NOSLIP) {
								// no slip boundary cell -> do reflection
								cells[current][i][j][k][l] = cells[other][i][j][k][inv];
							}
							else {
								// normal fluid cell 
								cells[current][i][j][k][l] = cells[other][i + e_x[inv]][j + e_y[inv]][k + e_z[inv]][l];
							}

						}


			// collide
#		pragma omp for  schedule(static)
			for (int i = 1; i < SIZE - 1; i++)
				for (int j = 1; j < SIZE - 1; j++)
					for (int k = 1; k < SIZE - 1; k++) {
						// skip boundaries during collision
						if (flags[i][j][k] == NOSLIP)
							continue;

						// turbulence model on?
						if (SMAGORINSKY) {
							collideSmagorinsky(cells[current][i][j][k], flags[i][j][k]);
							continue;
						}

						// standard collide without turbulence model
						Real rho = 0.0, u_x = 0.0, u_y = 0.0, u_z = 0.0;
						// normal fluid cell
						for (int l = 0; l < 19; l++) {
							const Real fi = cells[current][i][j][k][l];
							rho += fi;
							u_x += e_x[l] * fi;
							u_y += e_y[l] * fi;
							u_z += e_z[l] * fi;
						}

						if (flags[i][j][k] == VELOCITY) {
							// fixed velocity for LDC setup
							u_x = LDC_VELOCITY;
						}

						for (int l = 0; l < 19; l++) {
							const Real tmp = (e_x[l] * u_x + e_y[l] * u_y + e_z[l] * u_z);
							Real feq = w[l] * (rho -
								(3.0 / 2.0 * (u_x * u_x + u_y * u_y + u_z * u_z)) +
								(3.0 * tmp) +
								(9.0 / 2.0 * tmp * tmp));

							cells[current][i][j][k][l] =
								(1.0 - omega) * cells[current][i][j][k][l] +
								omega * feq;
						}
					} // ijk

		} // omp

		// We're done for one time step, switch the grid... 
		other = current;
		current = (current + 1) % 2;


		// picture output...

		// should we write an image for this step?
		if ((s % IMAGESTEPS) == 0) {
			writeImage(s);
		}
	} // STEPS

	printf("LBM-simulation done!\n");
	return 0;
}

// helper function to write out the velocity field
// or densities to an image
#define OUTNAME "outlbm"
void writeImage(int s) {
	// pressure or velocities?
	const bool showPressure = false;

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

	int k = (SIZE - 1) / 4; // specify the plane along the xy axis to write out
	const Real minDensity = 0.94; // manually set density range for picture
	const Real maxDensity = 1.095;
	const Real minVel = 0.;       // manually set velocity range for picture
	const Real maxVel = 0.1;
	Real totalMass = 0.; // this should be constant

	Real rhomin = 10.0, rhomax = -10.0;
	for (int j = SIZE - 1; j >= 0; j--) {
		for (int i = 0; i < SIZE; i++) {
			Real rho = 0.0, u_x = 0.0, u_y = 0.0, u_z = 0.0;
			for (int l = 0; l < 19; l++) {
				const Real fi = cells[other][i][j][k][l];
				rho += fi;
				totalMass += fi;
				u_x += e_x[l] * fi;
				u_y += e_y[l] * fi;
				u_z += e_z[l] * fi;
			}

			if (showPressure) {
				int grey = (int)((rho - minDensity) / (maxDensity - minDensity) * 255.0);
				if (grey < 0) grey = 0;
				if (grey > 255) grey = 255;
				fprintf(outfile, "%4d %4d %4d ", grey, grey, grey);
				if (rho > rhomax) rhomax = rho;
				if (rho < rhomin) rhomin = rho;
			}
			else {
				// show velocities
				// use absolute values
				if (u_x < 0.) u_x *= -1.;
				if (u_y < 0.) u_y *= -1.;
				if (u_z < 0.) u_z *= -1.;
				// to RGB...
				int r = (int)((u_x - minVel) / (maxVel - minVel) * 255.0);
				if (r < 0) r = 0;
				if (r > 255) r = 255;
				int g = (int)((u_y - minVel) / (maxVel - minVel) * 255.0);
				if (g < 0) g = 0;
				if (g > 255) g = 255;
				int b = (int)((u_z - minVel) / (maxVel - minVel) * 255.0);
				if (b < 0) b = 0;
				if (b > 255) b = 255;
				fprintf(outfile, "%4d %4d %4d ", r, g, b);
			}
		}
		fprintf(outfile, "\n");
	}

	fclose(outfile);
	printf("Wrote %s, min=%f max=%f. total mass=%f\n", outfilename, rhomin, rhomax, totalMass);
	// image written...		

	// try to convert, assumes you have ImageMagick
	char convert[256];
	sprintf(convert, "convert " OUTNAME "_%04d.ppm " OUTNAME "_%04d.png ", s, s);
	system(convert);
}


