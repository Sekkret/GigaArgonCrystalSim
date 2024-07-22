/**
 * program: Giga Argon Molecular Dynamics Simulation
 * author: Bartlomiej Baur
 * works on ssh68
 * compilation: nvcc argon.cu CFG/cfg_parse.c -o argon -lm -lcudart --expt-relaxed-constexpr
**/
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <time.h>
#include "CFG/cfg_parse.h" //Simple library to manage properties files.
#include "CFG/cfg_parse.c"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h> //rand function to use in kernels

/**
 * CUDA Error Handling
**/
#define CUDACHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
	    if (error_code != cudaSuccess)
	    {
			fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
			fflush(stderr);
			exit(error_code);
	    }
}

/**
 * For some reason atomicAdd(double*, double) avaiable now is not visible.
 *This is an AI generated implementation of this overload.
**/
__device__ double atomicAddDouble(double* address, double val) {
	    unsigned long long int* address_as_ull = (unsigned long long int*)address;
	    unsigned long long int old = *address_as_ull, assumed;
	    do {
	        assumed = old;
	        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	    } while (assumed != old);
		    return __longlong_as_double(old);
}

/**
 * constants:
 * epsilon - dielectric constant
 * k - boltzmann constant
 **/
__device__ const double epsilon = 1; // [kJ/mol]
__device__ const double k = 8.31e-3; // [kJ/(K*mol)]

/**
 * Load parameters of the simulation:
 * n - number of atoms in the edge of the crystal,
 * e - epsilon i.e. relative permittivity,
 * R - distance, for which Van der Vaals potential is minimal,
 * f - spring constant of boundaries,
 * L - box size,
 * a - initial crystal lattice constant,
 * T0 - initial temperature,
 * tau - time step
 * 
 * Calculated parameters and hard-coded constants:
 * k - Boltzmann constant,
 * epsilon - dielectric constant,
 * N - total number of atoms,
**/
//device constants
__device__ int n, d_N; 
__device__ double m, e, R, f, L, a, T0, d_tau;
//host constants
int h_N, S, S_out, S_xyz;
double h_tau;

/**
 * function declarations
**/
int load_parameters(const char* param_file_); //loads parameters for the simulation from the given file
__global__ void r_init(double* r); //sets starting positions of the atoms in the crystals
__global__ void p_init(double* p); //generates random initial momenta based on the maxwell's velocity distribution
__global__ void calc_VFP(double* r, double* V, double* F, double* P); //calculates potential energy, forces in the system and pressure
__global__ void calc_EkET(double* p, double* V, double* Ek, double* E, double* T); // calculates kinetic and total energy
__global__ void update_p(double* p, double* F);
__global__ void update_rp(double* r, double* p, double* F);

int main(int argc, const char *argv[]) {

	// Program must be executed with 3 additional parameters.
	if(argc!=4) {
		printf("Uzycie:\n");
		printf("./argon <parametry> <wyjscie energii> <wyjscie wspolrzednych>\n");
		return -1;
	}
	
	// Load parameters from file to the device.
	if( load_parameters(argv[1]) == -1){
		printf("Nie udalo sie odczytac pliku konfiguracyjnego.");
		return -1;
	}

	// Prepare files to save the results.
	FILE *xyz, *out;
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

	//Initialize simulation data.
	printf("Inicjalizacja symulacji... ");
	/**
	 * Simulation data:
	 * r - positions of all atoms [3N],
	 * p - momenta of all atoms [3N],
	 * F - forces affecting atoms [3N],
	 * V - potential of the system,
	 * EK - total kinetic energy of the system,
	 * T - temperature of the system,
	 * P - pressure of the system,
	**/
	double *h_r = (double*) malloc(3*h_N*sizeof(double));
	double *d_r, *p, *F;
	double *V, *Ek, *E;
	double *T, *P;
	CUDACHECK( cudaMalloc(&d_r, 3*h_N*sizeof(double)) );
	CUDACHECK( cudaMalloc((void**)&p, 3*h_N*sizeof(double)) );
	CUDACHECK( cudaMalloc((void**)&F, 3*h_N*sizeof(double)) );
	CUDACHECK( cudaMallocManaged(&V, sizeof(double)) );
	CUDACHECK( cudaMallocManaged(&Ek, sizeof(double)) );
	CUDACHECK( cudaMallocManaged(&E, sizeof(double)) );
	CUDACHECK( cudaMallocManaged(&T, sizeof(double)) );
	CUDACHECK( cudaMallocManaged(&P, sizeof(double)) );

	r_init<<<1,1>>>(d_r);
	p_init<<<1,1>>>(p);
	cudaDeviceSynchronize();
	dim3 blockDims(32,32); //one block can have up to 1024 threads.
	dim3 gridDims(h_N/32+1,h_N/32+1);
	calc_VFP<<<gridDims, blockDims>>>(d_r, V, F, P);
	cudaDeviceSynchronize();
	calc_EkET<<<1,3*h_N>>>(p, V, Ek, E, T);
	cudaDeviceSynchronize();

	//saving initial configuration of atoms
	cudaMemcpy(h_r, d_r, 3*h_N*sizeof(double), cudaMemcpyDeviceToHost);
	fprintf(xyz, "%d\n\n", h_N);
	for(int i=0; i<h_N; i++) {
		fprintf(xyz, "Ar ");
		for(int j=0; j<3; j++) {
			fprintf(xyz, "%f ", h_r[3*i+j]);
		}
		fprintf(xyz, "\n");
	}
	//saving initial simulation parameters' values
	fprintf(out, "0\t\t%f\t\t%f\t\t%f\t\t%f\t\t%f\n", *E, *Ek, *V, *T, *P);

	printf("Gotowe!\n");

	//main simulation loop
	printf("Rozpoczeto symulacje...\n");
	for(int step=1; step<=S; step++) {
		
		update_rp<<<1,3*h_N>>>(d_r, p, F);
		cudaDeviceSynchronize();
		calc_VFP<<<gridDims, blockDims>>>(d_r, V, F, P);

		//print out positions if necessary
		if(step%S_xyz==0) {
			cudaMemcpy(h_r, d_r, 3*h_N*sizeof(double), cudaMemcpyDeviceToHost);
			fprintf(xyz, "%d\n\n", h_N);
			for(int i=0; i<h_N; i++) {
				fprintf(xyz, "Ar ");
				for(int j=0; j<3; j++) {
					fprintf(xyz, "%f ", h_r[3*i+j]);
				}
				fprintf(xyz, "\n");
			}
		}

		cudaDeviceSynchronize();
		update_p<<<1,3*h_N>>>(p, F);
		cudaDeviceSynchronize();
		
		//print out calculation results if necessary
		if(step%S_out==0) {
			calc_EkET<<<1,3*h_N>>>(p, V, Ek, E, T);
			cudaDeviceSynchronize();
			fprintf(out, "%f\t\t%f\t\t%f\t\t%f\t\t%f\t\t%f\n", h_tau*step, *E, *Ek, *V, *T, *P);
		}
	}
	printf("Obliczenia zakonczone!\n");

	//cleaning up
	fclose(xyz);
	fclose(out);
	cudaFree(d_r);
	free(h_r);
	cudaFree(p);
	cudaFree(V);
	cudaFree(Ek);
	cudaFree(E);
	cudaFree(T);
	cudaFree(P);

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
	double tmp_d;
	int tmp_i;
	
	tmp_i = atoi( cfg_get(cfg, "n") ); //now tmp_i is n
	cudaMemcpyToSymbol(n, &tmp_i, sizeof(int)); 

	h_N = tmp_i * tmp_i * tmp_i; //N=n*n*n; N is on the host!
	cudaMemcpyToSymbol(d_N, &h_N, sizeof(int)); 
	
	tmp_d = strtod( cfg_get(cfg, "a") , NULL); //now tmp_d is a
	cudaMemcpyToSymbol(a, &tmp_d, sizeof(double)); 

	tmp_d = 1.22 * tmp_i * tmp_d; //L = 1.22 * a * n
	cudaMemcpyToSymbol(L, &tmp_d, sizeof(double));

	tmp_d = strtod( cfg_get(cfg, "m") , NULL); 
	cudaMemcpyToSymbol(m, &tmp_d, sizeof(double));

	tmp_d = strtod( cfg_get(cfg, "e") , NULL);
	cudaMemcpyToSymbol(e, &tmp_d, sizeof(double));

	tmp_d = strtod( cfg_get(cfg, "R") , NULL);
	cudaMemcpyToSymbol(R, &tmp_d, sizeof(double));
	
	tmp_d = strtod( cfg_get(cfg, "f") , NULL);
	cudaMemcpyToSymbol(f, &tmp_d, sizeof(double));

	tmp_d = strtod( cfg_get(cfg, "T_0") , NULL);
	cudaMemcpyToSymbol(T0, &tmp_d, sizeof(double));

 	h_tau = strtod( cfg_get(cfg, "tau") , NULL);
	cudaMemcpyToSymbol(d_tau, &h_tau, sizeof(double));


	// variables on the host:
	S = atoi( cfg_get(cfg, "S") );
	S_out = atoi( cfg_get(cfg, "S_out") );
	S_xyz = atoi( cfg_get(cfg, "S_xyz") );
	
	cfg_free(cfg);
	printf("Zaladowano parametry z pliku %s\n", param_file_);
	return 0;
}  

__global__
void r_init(double *r) {
	/**
	 * This function generates positions of atoms in crystall.
	 * Vectors b0, b1, and b2 are lattice vectors.
	 * The lattice constant a is defined in simulation parameters.
	 * The code is executed on the device mainly to avoid data copying.
	 * Parallelization of initialization is not necessary.
	 **/
	//int index = blockIdx.x * blockDim.x + threadIdx.x;
	//int stride = blockDim.x * gridDim.x;

	double b0[3] = {a, 0, 0};
	double b1[3] = { a/2, a*sqrt(3)/2, 0 };
	double b2[3] = { a/2, a*sqrt(3)/6, a*sqrt(2.0/3) };

	int I = 0;
	for(int i2=0; i2<n; i2++)
	for(int i1=0; i1<n; i1++)
	for(int i0=0; i0<n; i0++)
		for(int j=0; j<3; j++) {
			//r is a global array stored on the device.
		 	r[I] = (i0-(n-1)/2) * b0[j] + (i1-(n-1)/2) * b1[j] + (i2-(n-1)/2) * b2[j];
			I++;
		}
}

__global__
void p_init(double *p) {

	curandState state;
	curand_init(clock64(), 0, 0, &state);

	//value of kinetic energy and sign of momentum are randomly selected
	for(int i=0; i<3*d_N; i++) {
		double lambda = curand_uniform(&state);
		double E = -0.5 * k * T0 * log(lambda);

		int signum = ( curand_uniform(&state) >= 0 ) ? 1 : -1;
		p[i] = signum * sqrt(2*m*E);
	}

	//momentum is normalized, so the total momentum of the system is zero.
	double P[3] = {0,0,0};
	for(int i=0; i<d_N; i++) {
		for(int j=0; j<3; j++) {
			P[j]+=p[3*i+j];
		}
	}
		P[0]/=d_N;
		P[1]/=d_N;
		P[2]/=d_N;
	for(int i=0; i<d_N; i++) {
		for(int j=0; j<3; j++) {
			p[3*i+j] -= P[j];
		}
	}

	// scale the momenta, so they will mach to the temperature T0
	double T = 0;
	for(int i=0; i<3*d_N; i++) {
		T += p[i]*p[i]; //p^2
	}
	T /= 2*m; // E_k
	T = 2 * T / 3 / d_N / k; // actual T
	double scale_factor = sqrt(T0 / T);
    for(int i = 0; i < 3 * d_N; i++) {
        p[i] *= scale_factor;
    }

}

__global__
void calc_VFP(double* r, double* V, double* F, double* P) {
	/**
	 * Function calculates forces affecting every particle, their total potential energy and pressure in the system.
	 * Every atom is affected by every other atom and the "wals" potential - invisible box.
	 * Forces that originates from the box generates pressure.
	**/

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int stride_x = blockDim.x * gridDim.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_y = blockDim.y * gridDim.y;
	
	double dr, tmp1, tmp2;
	double local_V = 0;
	double local_P = 0;
	double local_F[3] = {0,0,0};
	
	// all variables of interest are set to zero in order to perform summation
	if(index_x == 0 && index_y == 0) {
		*P = 0;
		*V = 0;
		for(int i=0; i<3*d_N; i++)
			F[i] = 0;
	}

	__syncthreads();

	for(int i=index_x; i<d_N; i+=stride_x) {
		for(int j=index_y; j<i; j+=stride_y) {
			dr = sqrt( (r[3*i]-r[3*j])*(r[3*i]-r[3*j]) + (r[3*i+1]-r[3*j+1])*(r[3*i+1]-r[3*j+1]) + (r[3*i+2]-r[3*j+2])*(r[3*i+2]-r[3*j+2]) ); // Here, dr is square of a distance between atoms.
			tmp1 = R / dr;
			tmp2 = tmp1 * tmp1 * tmp1 * tmp1 * tmp1 * tmp1; // (R/r)^6
			local_V += epsilon * (  tmp2*tmp2 - 2*tmp2  ); // Van der Vaals potential between two atoms contributes to total potential.
			for(int k=0; k<3; k++) { // and now: Van der Vaals forces.
				local_F[k] += 12 * epsilon * (tmp2*tmp2 - tmp2) * (r[3*i+k]-r[3*j+k]) / dr / dr; //k-th component of force vector between i-th and j-th atom.
			}

			for(int k=0; k<3; k++) {
				atomicAddDouble(F+3*i+k, local_F[k]);
				atomicAddDouble(F+3*j+k, -local_F[k]);
			}	
		}
	}

	//Here are calculated elastic forces of the walls.
	if(index_x == 0) {
		for(int i=index_y; i<d_N; i+=stride_y) {

			dr = sqrt( r[3*i]*r[3*i] + r[3*i+1]*r[3*i+1] + r[3*i+2]*r[3*i+2] ); //Here, dr is the distance from the center
			tmp1 = dr<L ? 0 : 0.5*f*(dr-L)*(dr-L); //If an atom is too far away from the center, an elastic force occurs. This is its potential energy.
			local_V += tmp1; //the individual atom-wall potential contribute to total potential energy.
			for(int j=0; j<3; j++) { //finally - elastic force is calculated
					local_F[j] += dr<L ? 0 : f*(L-dr)*r[3*i+j]/dr;
			}
			local_P +=  sqrt( local_F[0]*local_F[0] + local_F[1]*local_F[1] + local_F[2]*local_F[2] ); // total force contribute to pressure too. P will be divided by area later.
			local_P /= 4*M_PI*L*L; // P is divided by box surface area and it's now a pressure.
		}

		for(int i = index_y; i<d_N; i+=stride_y) {
			for(int k=0; k<3; k++) {
				atomicAddDouble(F+3*i+k, local_F[k]);
			}
		}
	}
	atomicAddDouble(V, local_V);
	atomicAddDouble(P, local_P);
}

__global__
void calc_EkET(double* p, double* V, double* Ek, double* E, double* T) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	//kinetic energy
	double local_Ek=0;
	if(index == 0){
		*Ek = 0;
	}
	for(int i=index; i<3*d_N; i+=stride) {
		local_Ek += p[i]*p[i];
	}
	atomicAddDouble(Ek, local_Ek);
	
	__syncthreads();
	if(index == 0){
		*Ek /= 2*m;
		//total energy
		*E = *Ek + *V;
		*T = 2 * *Ek / 3 / d_N / k;
	}
}

__global__
void update_p(double* p, double* F) {
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i=index; i<3*d_N; i+=stride) {
		p[i] = p[i] + 0.5*F[i]*d_tau;
	}
}

__global__ 
void update_rp(double* r, double* p, double* F) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i=index; i<3*d_N; i+=stride) {
		p[i] = p[i] + 0.5*F[i]*d_tau;
		r[i] = r[i] + p[i]*d_tau/m;
	}
}
