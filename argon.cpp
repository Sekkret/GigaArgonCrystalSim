#include <iostream> 
#include <cmath>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include "../lib/SCF/config_file.h"

//parametry programu
int n, N, S, S_out, S_xyz;
double m, e, R, f, L, a, T0, tau;
const double epsilon = 1; // kJ/mol
const double k = 8.31e-3; // kJ/(K*mol)


//parametry symulacji
std::vector<double> r, p, F, Eki;
double V=0, Ek=0, E;
double T, P;
double VP=0, VS=0, FP=0, FS=0;

//zmienne pomocnicze
double dr;
double tmp1, tmp2, tmp3;

void load_parameters(const char* param_file_) {
	std::ifstream param;
	param.open(param_file_);
	std::vector<std::string> ln = {"n", "m", "e", "R", "f", "a", "T_0", "tau", "S", "S_out", "S_xyz"};
	CFG::ReadFile(param, ln, n, m, e, R, f, a, T0, tau, S, S_out, S_xyz);
	param.close();
	N = n*n*n;
	L = 1.22 * a * n;
	std::cout<<"Zaladowano parametry z pliku "<<param_file_<<std::endl;
}

void r_init() {
	//inicjalizacja polozen
	const std::vector<double> b0{ a, 0, 0 };
	const std::vector<double> b1{ a/2, a*sqrt(3)/2, 0 };
	const std::vector<double> b2{ a/2, a*sqrt(3)/6, a*sqrt(2.0/3) };

	for(int i2=0; i2<n; i2++)	
	for(int i1=0; i1<n; i1++)	
	for(int i0=0; i0<n; i0++)
		for(int j=0; j<3; j++) {
			double tmp = (i0-(n-1)/2)*b0[j] + (i1-(n-1)/2)*b1[j] + (i2-(n-1)/2)*b2[j];
			r.push_back(tmp);
		}
}

void p_init() {
	//losujemy Ek oraz znak pedu, nastepnie liczymy ped
	for(int i=0; i<3*N; i++) {
		double lambda = (double) (std::rand()%10000)/10000;
		double E = -0.5 * k * T0 * log(lambda);
		int signum = (std::rand()%2) % 2 == 0 ? 1 : -1;
		p.push_back(signum * sqrt(2*m*E));
	}
	//normujemy ped tak, aby dla calego ukladu wynosil zero
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
			p.at(3*i+j)-=P[j];
		}
	}
}

void calc_EkandE() {

	//energia kinetyczna
	Ek=0;
	for(int i=0; i<N; i++) {
		for(int j=0; j<3; j++) {
			Eki[i] = p[3*i+j]*p[3*i+j];
			Ek += Eki[i];
		}
	}
	Ek /= 2*m;

	//energia calkowita
	E = Ek + V;
}

void calc_VFP() {

	P = 0;
	V = 0;
	for(int i=0; i<3*N; i++)
		F[i] = 0;

	for(int i=0; i<N; i++) {

		dr = sqrt( r[3*i]*r[3*i] + r[3*i+1]*r[3*i+1] + r[3*i+2]*r[3*i+2] ); //odleglosc od srodka ukladu odniesienia
		tmp1 = dr<L ? 0 : 0.5*f*(dr-L)*(dr-L); //potencjal od scianek VS
		V += tmp1;
		for(int j=0; j<3; j++) { //sila od scianek FS
		       	F[3*i+j] += dr<L ? 0 : f*(L-dr)*r[3*i+j]/dr;
		}
		P += sqrt( F[3*i]*F[3*i] + F[3*i+1]*F[3*i+1] + F[3*i+2]*F[3*i+2] ); //akumulacja cisnienia. Bedzie ono jeszcze mnozone przez wspolczynnik!
		for(int j=0; j<i; j++) {
			dr = sqrt( (r[3*i]-r[3*j])*(r[3*i]-r[3*j]) + (r[3*i+1]-r[3*j+1])*(r[3*i+1]-r[3*j+1]) + (r[3*i+2]-r[3*j+2])*(r[3*i+2]-r[3*j+2]) ); //od teraz dr to odleglosc miedzy dwoma atomami!!!
			tmp1 = R/dr; //rachunek pomocniczy
			tmp2 = tmp1 * tmp1 * tmp1 * tmp1 * tmp1 * tmp1; //rachunek pomocniczy
			tmp3 = epsilon * (  tmp2*tmp2 - 2*tmp2  ); //potencjal miedzy atomami VP
			V += tmp3;
			for(int k=0; k<3; k++) { //sily van der Waalsa FP
				tmp3 = 12 * epsilon * (tmp2*tmp2 - tmp2) * (r[3*i+k]-r[3*j+k]) / dr / dr; //k-ta skladowa sily pomiedzy i-tym i j-tym atomem
				F[3*i+k] += tmp3;
			       	F[3*j+k] += -tmp3;
			}
		}
	}
	P /= 4*M_PI*L*L; //konczenie rachunkow P

}


int main(int argc, const char *argv[]) {

	//inicjalizacja programu
	if(argc!=4) {
		std::cout<<"Uzycie:"<<std::endl;
		std::cout<<"argon <parametry> <wyjscie energii> <wyjscie wspolrzednych>"<<std::endl;
		return -1;
	}
	srand( time( NULL ) );
	load_parameters(argv[1]);

	std::ofstream xyz, out;
	xyz.open(argv[3]);
	out.open(argv[2]);
	std::cout<<"Otworzono plik "<<argv[2]<<" do zapisu danych wyjsciowych programu."<<std::endl;
	std::cout<<"Otworzono plik "<<argv[3]<<" do zapisu wspolrzednych symulowanych atomow."<<std::endl;

	out<<"t\t\tE\t\tEk\t\tV\t\tT\t\tP"<<std::endl;
	std::cout<<"OK"<<std::endl;

	//inicjalizacja symulacji
	std::cout<<"Inicjalizacja symulacji... ";
	r_init();
	p_init();
	F.reserve(3*N);
	Eki.reserve(N);
	calc_VFP();
	calc_EkandE();

	//wypisanie poczatkowych polozen
	xyz<<N<<std::endl<<std::endl;
	for(int i=0; i<N; i++) {
		xyz<<"Ar ";
		for(int j=0; j<3; j++) {
			xyz<<r[3*i+j]<<" ";
		}
		xyz<<std::endl;
	}
	//zapisanie sytuacji poczatkowej do OUT.txt
	out<<0<<"\t\t"<<E<<"\t\t"<<Ek<<"\t\t"<<V<<"\t\t"<<T0<<"\t\t"<<P<<std::endl;

	std::cout<<"OK"<<std::endl;



	std::cout<<"Rozpoczeto symulacje..."<<std::endl;
	//petla symulacji
	for(int step=1; step<=S; step++) {
		
		for(int i=0; i<3*N; i++) {
			p[i] = p[i] + 0.5*F[i]*tau;
			r[i] = r[i] + p[i]*tau/m;
		}
		calc_VFP();
		for(int i=0; i<3*N; i++) {
		       p[i] = p[i] + 0.5*F[i]*tau;
		}	       
		calc_EkandE();
		T = 2*Ek/3/N/k;


		if(step%S_out==0) {
			out<<tau*step<<"\t\t"<<E<<"\t\t"<<Ek<<"\t\t"<<V<<"\t\t"<<T<<"\t\t"<<P<<std::endl;
		}
		if(step%S_xyz==0) {
			xyz<<N<<std::endl<<std::endl;
			for(int i=0; i<N; i++) {
				xyz<<"Ar"<<" ";
				for(int j=0; j<3; j++) {
					xyz<<r[3*i+j]<<" ";
				}
				xyz<<std::endl;
			}
		}
	}
	std::cout<<"Obliczenia zakonczone!"<<std::endl;

	xyz.close();
	out.close();
	return 0;
}
