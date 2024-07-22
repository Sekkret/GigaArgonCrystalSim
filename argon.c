/**
 * program: Giga Argon Molecular Dynamics Simulation
 * author: Bartlomiej Baur
 * compilation: gcc argon.c CFG/cfg_parse.c -o argon -lm
**/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "CFG/cfg_parse.h" //Simple library to manage properties files.

// program parameters see README.md for reference
int n, N, S, S_out, S_xyz;
double m, e, R, f, L, a, T0, tau;
const double epsilon = 1; // [kJ/mol]
const double k = 8.31e-3; // [kJ/(K*mol)]

// simulation parameters
double* r, *p, *F, *Eki;
double V=0, Ek=0, E;
double T, P;
double VP=0, VS=0, FP=0, FS=0;

// auxiliary variables
double dr;
double tmp1, tmp2, tmp3;

// function declarations
int load_parameters(const char* param_file_); //loads parameters for the simulation from the given file
void r_init(); //sets starting positions of the atoms in the crystals
void p_init(); //generates random initial momenta based on the maxwell's velocity distribution
void calc_EkandE(); // calculates kinetic and total energy
void calc_VFP(); //calculates potential energy, forces in the system and pressure

int main(int argc, const char *argv[]) {

	//program initialization
	//Chceck if the the program is executing with 3 additional parameters.
	if(argc!=4) {
		printf("Uzycie:\n");
		printf("argon <parametry> <wyjscie energii> <wyjscie wspolrzednych>\n");
		return -1;
	}
	srand( time( NULL ) ); //seed for random numbers generator
	if( load_parameters(argv[1]) == -1) //read parameters from given file
		return -1;

	//prepare files to save the results:
	FILE* xyz, *out;
	if(xyz=fopen(argv[3], "w")) {
		printf("Otworzono plik %s do zapisu wspolrzednych symulowanych atomow\n", argv[3]);
	}
	else{
		printf("Nie mozna otworzyc pliku %s do zapisu!", argv[3]);
		return 1;
	}
	if( out=fopen(argv[2], "w") ) {
		printf("Otworzono plik %s do zapisu danych wyjsciowych programu\n", argv[2]);
	}
	else{
		printf("Nie mozna otworzyc pliku %s do zapisu!", argv[2]);
		return 1;
	}

	fprintf(out, "t\t\tE\t\tEk\t\tV\t\tT\t\tP\n");

	printf("Inicjalizacja symulacji... ");
	// allocating memory for arrays
	r = (double*) malloc(3*N*sizeof(double));
	p = (double*) malloc(3*N*sizeof(double));
	F = (double*) malloc(3*N*sizeof(double));
	Eki = (double*) malloc(N*sizeof(double));
	//simulation initialization
	r_init();
	p_init();
	calc_VFP();
	calc_EkandE();
	printf("Gotowe!\n");

	//saving initial configuration of atoms
	fprintf(xyz, "%d\n\n", N);
	for(int i=0; i<N; i++) {
		fprintf(xyz, "Ar ");
		for(int j=0; j<3; j++) {
			fprintf(xyz, "%f ", r[3*i+j]);
		}
		fprintf(xyz, "\n");
	}
	//saving initial simulation parameters' values
	fprintf(out, "0\t\t%f\t\t%f\t\t%f\t\t%f\t\t%f\n", E, Ek, V, T0, P);

	printf("OK\n");




	printf("Rozpoczeto symulacje...\n");
	//main simulation loop
	for(int step=1; step<=S; step++) {
		
		for(int i=0; i<3*N; i++) { //update momenta and positions
			p[i] = p[i] + 0.5*F[i]*tau;
			r[i] = r[i] + p[i]*tau/m;
		}
		//calculate new energies, forces and finish calculating new momenta
		calc_VFP();
		for(int i=0; i<3*N; i++) {
		       p[i] = p[i] + 0.5*F[i]*tau;
		}	       
		calc_EkandE();
		T = 2*Ek/3/N/k;


		//print out calculation results if necessary
		if(step%S_out==0) {
			fprintf(out, "%f\t\t%f\t\t%f\t\t%f\t\t%f\t\t%f\n", tau*step, E, Ek, V, T, P);
		}
		if(step%S_xyz==0) {
			fprintf(xyz, "%d\n\n", N);
			for(int i=0; i<N; i++) {
				fprintf(xyz, "Ar ");
				for(int j=0; j<3; j++) {
					fprintf(xyz, "%f ", r[3*i+j]);
				}
				fprintf(xyz, "\n");
			}
		}
	}
	printf("Obliczenia zakonczone!\n");

	//closing output files
	fclose(xyz);
	fclose(out);
	//clearing out memory
	free(r);
	free(p);
	free(F);
	free(Eki);
	return 0;
}

int load_parameters(const char* param_file_) {
	struct cfg_struct *cfg;
	cfg = cfg_init();
	if (cfg_load(cfg, param_file_) < 0)
	{
	  fprintf(stderr,"Unable to load %s\n", param_file_);
	  return -1;
	}
	n = atoi( cfg_get(cfg, "n") );
	m = strtod( cfg_get(cfg, "m") , NULL);
	e = strtod( cfg_get(cfg, "e") , NULL);
	R = strtod( cfg_get(cfg, "R") , NULL);
	f = strtod( cfg_get(cfg, "f") , NULL);
	a = strtod( cfg_get(cfg, "a") , NULL);
	T0 = strtod( cfg_get(cfg, "T_0") , NULL);
	tau = strtod( cfg_get(cfg, "tau") , NULL);
	S = atoi( cfg_get(cfg, "S") );
	S_out = atoi( cfg_get(cfg, "S_out") );
	S_xyz = atoi( cfg_get(cfg, "S_xyz") );
	
	cfg_free(cfg);
	
	N = n*n*n;
	L = 1.22 * a * n;
	printf("Zaladowano parametry z pliku %s\n", param_file_);


	return 0;
}

void r_init() {
	/**
	 * This function generates positions of atoms in crystall.
	 * Vectors b0, b1, and b2 are lattice vectors.
	 * The lattice constant a is defined in simulation parameters.
	 **/
	double b0[3] = {a, 0, 0};
	double b1[3] = { a/2, a*sqrt(3)/2, 0 };
	double b2[3] = { a/2, a*sqrt(3)/6, a*sqrt(2.0/3) };

	int I = 0;
	for(int i2=0; i2<n; i2++)
	for(int i1=0; i1<n; i1++)
	for(int i0=0; i0<n; i0++)
		for(int j=0; j<3; j++) {
		 	r[I] = (i0-(n-1)/2) * b0[j] + (i1-(n-1)/2) * b1[j] + (i2-(n-1)/2) * b2[j];
			I++;
		}
}

void p_init() {
	//value of kinetic energy and sign of momentum are randomly selected
	for(int i=0; i<3*N; i++) {
		double lambda = (double) (rand()%10000)/10000;
		double E = -0.5 * k * T0 * log(lambda);
		int signum = (rand()%2) % 2 == 0 ? 1 : -1;
		p[i] = signum * sqrt(2*m*E);
	}
	//momentum is normalized, so the total momentum of the system is zero.
	double P[3] = {0,0,0};
	for(int i=0; i<N; i++) {
		for(int j=0; j<3; j++) {
			P[j]+=p[3*i+j];
		}
	}
	P[0]/=N;
	P[1]/=N;
	P[2]/=N;
	for(int i=0; i<N; i++) {
		for(int j=0; j<3; j++) {
			p[3*i+j] -= P[j];
		}
	}
}

void calc_EkandE() {

	//kinetic energy
	Ek=0;
	for(int i=0; i<N; i++) {
		for(int j=0; j<3; j++) {
			Eki[i] = p[3*i+j]*p[3*i+j];
			Ek += Eki[i];
		}
	}
	Ek /= 2*m;

	//total energy
	E = Ek + V;
}

void calc_VFP() {
	/**
	 * Function calculates forces affecting every particle, their total potential energy and pressure in the system.
	 * Every atom is affected by every other atom and the "wals" potential - invisible box.
	 * Forces that originates from the box generates pressure.
	**/
	
	// all variables of interest are set to zero in order to perform summation
	P = 0;
	V = 0;
	for(int i=0; i<3*N; i++)
		F[i] = 0;

	for(int i=0; i<N; i++) {

		dr = sqrt( r[3*i]*r[3*i] + r[3*i+1]*r[3*i+1] + r[3*i+2]*r[3*i+2] ); //Here, dr is the distance from the center
		tmp1 = dr<L ? 0 : 0.5*f*(dr-L)*(dr-L); //If an atom is too far away from the center, an elastic force occurs. This is its potential energy.
		V += tmp1; //the individual atom-wall potential contribute to total potential energy.
		for(int j=0; j<3; j++) { //finally - elastic force is calculated
		       	F[3*i+j] += dr<L ? 0 : f*(L-dr)*r[3*i+j]/dr;
		}
		P += sqrt( F[3*i]*F[3*i] + F[3*i+1]*F[3*i+1] + F[3*i+2]*F[3*i+2] ); // total force contribute to pressure too. P will be divided by area later.
		// in this loop we calculate atom-atom interactions.
		for(int j=0; j<i; j++) {
			dr = sqrt( (r[3*i]-r[3*j])*(r[3*i]-r[3*j]) + (r[3*i+1]-r[3*j+1])*(r[3*i+1]-r[3*j+1]) + (r[3*i+2]-r[3*j+2])*(r[3*i+2]-r[3*j+2]) ); //we re-use the same variable. Now it's a distance between atoms!
			tmp1 = R / dr;
			tmp2 = tmp1 * tmp1 * tmp1 * tmp1 * tmp1 * tmp1;
			tmp3 = epsilon * (  tmp2*tmp2 - 2*tmp2  ); // Van der Vaals potential between two atoms...
			V += tmp3; // ... also contributes to total potential
			for(int k=0; k<3; k++) { // and now: Van der Vaals forces.
				tmp3 = 12 * epsilon * (tmp2*tmp2 - tmp2) * (r[3*i+k]-r[3*j+k]) / dr / dr; //k-th component of force vector between i-th and j-th atom.
				F[3*i+k] += tmp3;
			    F[3*j+k] += -tmp3;
			}
		}
	}
	P /= 4*M_PI*L*L; // P is divided by box surface area and it's now a pressure.
}
