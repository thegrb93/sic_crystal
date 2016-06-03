#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/StdVector>

#ifdef _OPENMP
inline int my_omp_get_thread_num(){ return omp_get_thread_num(); }
inline int my_omp_get_num_threads(){ return omp_get_num_threads(); };
#else
inline int my_omp_get_thread_num(){ return 0; }
inline int my_omp_get_num_threads(){ return 1; };
#endif

enum : int {
	ATOM_SI = 0,
	ATOM_C = 1
};

struct PotentialParameters_TwoBody
{
	int n;
	double H, D, Z, W, dVdR, Vr;
};
struct PotentialParameters_ThreeBody
{
	double B, Theta_bar, C, Gamma;
};

typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> Eigen3dVector;

struct Element_Atom
{
	int type;
	Eigen::Vector3d position, velocity, analytical_force, numeric_force;

	Eigen3dVector distances;
	std::vector <double> distance_lengths;
	
	friend std::ostream& operator<< (std::ostream &out, Element_Atom &atom);
	friend std::istream& operator>> (std::istream &in, Element_Atom &atom);

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::vector<Element_Atom, Eigen::aligned_allocator<Element_Atom>> ElementVector;

struct Crystal_SiC
{
	double Atomic_Masses[2];
	//Particles
	ElementVector m_particles;
	//Length of the crystal
	double m_length;
	double m_mass;
	int m_start;
	int m_end;
	PotentialParameters_TwoBody m_twobody_constants[3];
	PotentialParameters_ThreeBody m_threebody_constants;

	void(*stepCallback)(int);
	void(*calcDistanceCallback)();
	void(*calcForcesCallback)();

	Crystal_SiC();
	void calc2BodyEnergyForces();
	void calc3BodyEnergyForces();
	void findRings();
	void randomizeVelocity(double temp);
	void regulateTemperature(double temp);
	void simulate(double timestep, int steps);
	void saveFile(const std::string& filen);
	int loadFile(const char* filen);
};
