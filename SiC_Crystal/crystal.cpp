#include "crystal.h"
#include <flann/flann.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <cmath>

const double lambda = 5.0;
const double epsilon = 3.0;
const double permitivity = 14.4;

const double r_0 = 2.9;
const double r_c = 7.35;

const double boltzmann_constant = 8.617332478e-5; //eV/K

inline double SiCTwoBodyPotential(const PotentialParameters_TwoBody& param, double r_ij)
{
	return param.H / std::pow(r_ij, param.n) +
		param.Z * permitivity * std::exp(-r_ij / lambda) / r_ij -
		param.D * permitivity * std::exp(-r_ij / epsilon) / 2 / std::pow(r_ij, 4) -
		param.W / std::pow(r_ij, 6);
}

inline double SiCTwoBodyPotentialShifted(const PotentialParameters_TwoBody& param, double r_ij)
{
	return SiCTwoBodyPotential(param, r_ij) - param.Vr - (r_ij - r_c) * param.dVdR;
}

inline double SiCTwoBodydVdR(const PotentialParameters_TwoBody& param, double r_ij)
{
	return -param.n * param.H / std::pow(r_ij, param.n + 1) -
		param.Z * permitivity * std::exp(-r_ij / lambda) * (std::pow(r_ij, -2) + 1.0 / r_ij / lambda) +
		param.D * permitivity / 2 * std::exp(-r_ij / epsilon) * (4 * epsilon + r_ij) / std::pow(r_ij, 5) / epsilon +
		6 * param.W / std::pow(r_ij, 7);
}

inline double SiCTwoBodydVdRShifted(const PotentialParameters_TwoBody& param, double r_ij)
{
	return SiCTwoBodydVdR(param, r_ij) - param.dVdR;
}

inline double SiCThreeBodyPotential(const PotentialParameters_ThreeBody& param, double r_ij, double r_ik, double costheta)
{
	return param.B * std::exp(param.Gamma * (1.0 / (r_ij - r_0) + 1.0 / (r_ik - r_0))) / (std::pow(costheta - param.Theta_bar, -2) + param.C);
}

inline double SiCThreeBodyPotentialP(const PotentialParameters_ThreeBody& param, double r_ij, double r_ik, double costheta)
{
	return 1.0 / (std::pow(costheta - param.Theta_bar, -2) + param.C);
}

inline double SiCThreeBodyPotentialR(const PotentialParameters_ThreeBody& param, double r_ij, double r_ik)
{
	return param.B * std::exp(param.Gamma * (1.0 / (r_ij - r_0) + 1.0 / (r_ik - r_0)));
}

inline double SiCThreeBodydPdX(const PotentialParameters_ThreeBody& param, double costheta)
{
	return (2*(costheta+1.0/3.0)/std::pow(param.C*std::pow(costheta+1.0/3.0,2)+1,2));
}

inline Eigen::Vector3d SiCThreeBodydRdX(const PotentialParameters_ThreeBody& param, double r_ij, double R, Eigen::Vector3d& v_ij)
{
	return v_ij * -(R * param.Gamma / r_ij / std::pow(r_ij-r_0, 2));
}

inline Eigen::Vector3d SiCThreeBodydCosdX(double r_ij, double r_ik, double costheta, Eigen::Vector3d& v_ij, Eigen::Vector3d& v_ik)
{
	return (v_ik / (r_ij*r_ik) - v_ij*(costheta / std::pow(r_ij, 2)));
}


Crystal_SiC::Crystal_SiC()
{
	m_twobody_constants[0] =
	{ // Silicon - Silicon
		7, //n
		23.67291, //H
		2.1636, //D
		1.442401, //Z
		0., //W
		0, //dVdR
		0  //Vr
	};
	m_twobody_constants[1] =
	{ // Silicon - Carbon
		9,
		447.09026,
		1.0818,
		-1.442401, //Z
		61.4694,
		0,
		0
	};
	m_twobody_constants[2] =
	{ // Carbon - Carbon
		7,
		471.74538,
		0.,
		1.442401, //Z
		0.,
		0,
		0
	};

	m_twobody_constants[0].dVdR = SiCTwoBodydVdR(m_twobody_constants[0], r_c);
	m_twobody_constants[1].dVdR = SiCTwoBodydVdR(m_twobody_constants[1], r_c);
	m_twobody_constants[2].dVdR = SiCTwoBodydVdR(m_twobody_constants[2], r_c);

	m_twobody_constants[0].Vr = SiCTwoBodyPotential(m_twobody_constants[0], r_c);
	m_twobody_constants[1].Vr = SiCTwoBodyPotential(m_twobody_constants[1], r_c);
	m_twobody_constants[2].Vr = SiCTwoBodyPotential(m_twobody_constants[2], r_c);

	m_threebody_constants =
	{
		9.003, // B
		-1.0/3.0, // cos_Theta_bar
		5.0, // C
		1.0 // Gamma
	};

	Atomic_Masses[0] = 1450.99768674;
	Atomic_Masses[1] = 1243.71230292;
	
	stepCallback = nullptr;
	calcDistanceCallback = nullptr;
	calcForcesCallback = nullptr;

	m_start = 0;
}

const double perterb = 1e-6;

//#define CalculateNumericalForce
void Crystal_SiC::calc2BodyEnergyForces()
{
	int end = m_end - 1;
	for (int i = m_start; i < end; ++i)
	{
		Element_Atom& particle_i = m_particles[i];

#		pragma omp parallel for
		for (int j = i + 1; j < m_particles.size(); ++j)
		{
			Element_Atom& particle_j = m_particles[j];

			Eigen::Vector3d distance = (particle_j.position - particle_i.position);
			typedef double(*RoundType)(double);
			distance -= (distance / m_length).unaryExpr((RoundType)round) * m_length;
			double length = distance.norm();

			if (length <= r_c)
			{
				Eigen::Vector3d distance_norm = distance / length;
				const PotentialParameters_TwoBody& param = m_twobody_constants[particle_i.type + particle_j.type];

				Eigen::Vector3d analytical_force = distance_norm * SiCTwoBodydVdRShifted(param, length);

				particle_i.distances[j] = distance;
				particle_j.distances[i] = -distance;
				particle_i.distance_lengths[j] = length;
				particle_j.distance_lengths[i] = length;

#				pragma omp critical
				{
					particle_i.analytical_force += analytical_force;
					particle_j.analytical_force -= analytical_force;
				}

#					ifdef CalculateNumericalForce
				Eigen::Vector3d numerical_force = distance_norm * ((SiCTwoBodyPotentialShifted(param, length + perterb) -
					SiCTwoBodyPotentialShifted(param, length - perterb)) / 2.0 / perterb);
				particle_i.numeric_force += numerical_force;
				particle_j.numeric_force -= numerical_force;
				total_energy += SiCTwoBodyPotentialShifted(param, length);
#					endif

			}
			else
			{
				particle_i.distance_lengths[j] = std::numeric_limits<double>::max();
				particle_j.distance_lengths[i] = std::numeric_limits<double>::max();
			}
		}
	}
}

void Crystal_SiC::calc3BodyEnergyForces()
{
	for (int i = m_start; i < m_end; ++i)
	{
		Element_Atom& particle_i = m_particles[i];
#		pragma omp parallel for
		for (int j = 0; j < m_particles.size(); ++j)
		{
			Element_Atom& particle_j = m_particles[j];
			if (i != j && particle_i.type != particle_j.type)
			{
				Eigen::Vector3d& vij = particle_i.distances[j];
				double rij = particle_i.distance_lengths[j];
				if (rij <= r_0)
				{
					for (int k = j + 1; k < m_particles.size(); ++k)
					{
						Element_Atom& particle_k = m_particles[k];
						if (i != k && particle_i.type != particle_k.type)
						{
							Eigen::Vector3d& vik = particle_i.distances[k];
							double rik = particle_i.distance_lengths[k];
							if (rik <= r_0)
							{
								double costheta = particle_i.distances[j].dot(particle_i.distances[k]) / rij / rik;
								double R = SiCThreeBodyPotentialR(m_threebody_constants, rij, rik);
								double P = SiCThreeBodyPotentialP(m_threebody_constants, rij, rik, costheta);
								double R_dpdx = R*SiCThreeBodydPdX(m_threebody_constants, costheta);

								Eigen::Vector3d dVdj = P*SiCThreeBodydRdX(m_threebody_constants, rij, R, vij) +
									R_dpdx*SiCThreeBodydCosdX(rij, rik, costheta, vij, vik);

								Eigen::Vector3d dVdk = P*SiCThreeBodydRdX(m_threebody_constants, rik, R, vik) +
									R_dpdx*SiCThreeBodydCosdX(rik, rij, costheta, vik, vij);

#								pragma omp critical
								{
									particle_i.analytical_force += (dVdj + dVdk);
									particle_j.analytical_force -= dVdj;
									particle_k.analytical_force -= dVdk;
								}

#								ifdef CalculateNumericalForce
								total_energy += R*P;
								Eigen::Vector3d numeric_force_ij(0, 0, 0);
								Eigen::Vector3d numeric_force_ik(0, 0, 0);
								for (int l = 0; l < 3; ++l)
								{
									Eigen::Vector3d v_perterb(0, 0, 0);
									v_perterb(l) = perterb;

									Eigen::Vector3d v_ij_p_1 = vij - v_perterb;
									Eigen::Vector3d v_ij_p_2 = vij + v_perterb;
									double pij1 = v_ij_p_1.norm();
									double pij2 = v_ij_p_2.norm();
									double costhetaij1 = v_ij_p_1.dot(vik) / pij1 / rik;
									double costhetaij2 = v_ij_p_2.dot(vik) / pij2 / rik;

									numeric_force_ij(l) = (SiCThreeBodyPotential(m_threebody_constants, pij1, rik, costhetaij1) -
										SiCThreeBodyPotential(m_threebody_constants, pij2, rik, costhetaij2)) / 2 / perterb;


									Eigen::Vector3d v_ik_p_1 = vik - v_perterb;
									Eigen::Vector3d v_ik_p_2 = vik + v_perterb;
									double pik1 = v_ik_p_1.norm();
									double pik2 = v_ik_p_2.norm();
									double costhetaik1 = v_ik_p_1.dot(vij) / pik1 / rij;
									double costhetaik2 = v_ik_p_2.dot(vij) / pik2 / rij;

									numeric_force_ik(l) = (SiCThreeBodyPotential(m_threebody_constants, rij, pik1, costhetaik1) -
										SiCThreeBodyPotential(m_threebody_constants, rij, pik2, costhetaik2)) / 2 / perterb;

								}

								particle_i.numeric_force -= (numeric_force_ij + numeric_force_ik);
								particle_j.numeric_force += numeric_force_ij;
								particle_k.numeric_force += numeric_force_ik;
#								endif
							}
						}
					}
				}
			}
		}
	}
}

const double ringThresholds[3] = {
	2.7 * 2.7, // Si - Si
	2.4 * 2.4, // Si - C
	1.9 * 1.9, // C - C
};
void Crystal_SiC::findRings()
{
	flann::Matrix<double> positions(new double[m_particles.size() * 3], m_particles.size(), 3);
	flann::Matrix<int> indices(new int[m_particles.size()], 1, m_particles.size());
	flann::Matrix<double> dists(new double[m_particles.size()], 1, m_particles.size());

#	pragma omp parallel for
	for (int i = 0; i < m_particles.size(); ++i)
	{
		memcpy(positions[i], m_particles[i].position.data(), sizeof(double) * 3);
	}
	flann::KDTreeSingleIndex<flann::L2<double>> index(positions, flann::KDTreeSingleIndexParams());
	index.buildIndex();

	std::vector<std::vector<int>> closest_particles(m_particles.size());
#	pragma omp parallel for
	for (int i = 0; i < m_particles.size(); ++i)
	{
		Element_Atom& particle_i = m_particles[i];
		flann::Matrix<double> position(particle_i.position.data(), 1, 3);
		int count = index.radiusSearch(position, indices, dists, ringThresholds[0], flann::SearchParams(-1));

		closest_particles[i].reserve(count);
		for (int j = 0; j < count; ++j)
		{
			Element_Atom& particle_j = m_particles[j];
			if (dists[0][j] <= ringThresholds[particle_i.type + particle_j.type])
			{
#				pragma omp critical
				{
					closest_particles[i].push_back(indices[0][j]);
				}
			}
		}
	}

	delete[] positions.ptr();
	delete[] indices.ptr();
	delete[] dists.ptr();
}

void Crystal_SiC::randomizeVelocity(double temp)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> rand_r(std::numeric_limits<double>::denorm_min(), 1);
	std::uniform_real_distribution<double> rand_theta(0, std::acos(-1)*2);

	double prefactors[2] = 
	{
		std::sqrt(temp * boltzmann_constant / Atomic_Masses[0]),
		std::sqrt(temp * boltzmann_constant / Atomic_Masses[1]),
	};

	Eigen::Vector3d total_momentum(0, 0, 0);

#	pragma omp parallel
	{
		int this_thread = my_omp_get_thread_num(), num_threads = my_omp_get_num_threads();
		size_t chunk_size = (m_particles.size() + num_threads - 1) / num_threads;
		ElementVector::iterator i = m_particles.begin() + this_thread * chunk_size;
		ElementVector::iterator my_end = m_particles.begin() + std::min<size_t>((this_thread + 1)*chunk_size, m_particles.size());
		for (; i != my_end; ++i)
		{
			double& prefactor = prefactors[i->type];
			i->velocity(0) = prefactor * std::sqrt(-2 * std::log(rand_r(gen))) * std::sin(rand_theta(gen));
			i->velocity(1) = prefactor * std::sqrt(-2 * std::log(rand_r(gen))) * std::sin(rand_theta(gen));
			i->velocity(2) = prefactor * std::sqrt(-2 * std::log(rand_r(gen))) * std::sin(rand_theta(gen));

#			pragma omp critical
			{
				total_momentum += i->velocity * Atomic_Masses[i->type];
			}
		}
	}

	Eigen::Vector3d shift = total_momentum / m_mass;
#	pragma omp parallel
	{
		int this_thread = my_omp_get_thread_num(), num_threads = my_omp_get_num_threads();
		size_t chunk_size = (m_particles.size() + num_threads - 1) / num_threads;
		ElementVector::iterator i = m_particles.begin() + this_thread * chunk_size;
		ElementVector::iterator my_end = m_particles.begin() + std::min<size_t>((this_thread + 1)*chunk_size, m_particles.size());
		for (; i != my_end; ++i)
		{
			i->velocity -= shift;
		}
	}
}

void Crystal_SiC::regulateTemperature(double temp)
{
	double kinetic_energy = 0;
#	pragma omp parallel
	{
		int this_thread = my_omp_get_thread_num(), num_threads = my_omp_get_num_threads();
		size_t chunk_size = (m_particles.size() + num_threads - 1) / num_threads;
		ElementVector::iterator i = m_particles.begin() + this_thread * chunk_size;
		ElementVector::iterator my_end = m_particles.begin() + std::min<size_t>((this_thread + 1)*chunk_size, m_particles.size());
		for (; i != my_end; ++i)
		{
			double energy = i->velocity.dot(i->velocity) * Atomic_Masses[i->type];
#			pragma omp atomic
			kinetic_energy += energy;
		}
	}
	double Tcurrent = kinetic_energy / double(3 * m_particles.size() - 3) / boltzmann_constant;
	double factor = std::sqrt(temp / Tcurrent);

#	pragma omp parallel
	{
		int this_thread = my_omp_get_thread_num(), num_threads = my_omp_get_num_threads();
		size_t chunk_size = (m_particles.size() + num_threads - 1) / num_threads;
		ElementVector::iterator i = m_particles.begin() + this_thread * chunk_size;
		ElementVector::iterator my_end = m_particles.begin() + std::min<size_t>((this_thread + 1)*chunk_size, m_particles.size());
		for (; i != my_end; ++i)
		{
			i->velocity *= factor;
			for (int o = 0; o < 3; ++o)
			{
				double& v = i->position(o);
				if (v < 0 || v > m_length)
				{
					v = std::fmod(v, m_length);
				}
			}
		}
	}
}

void Crystal_SiC::simulate(double timestep, int steps)
{
	//1500K/ns
	const double constant = -5.59615787e-7;

	for (int step = 0; step <= steps; ++step)
	{
		//double t = timestep * step;
		double temp = step >= 1000000 ? 2000 : 3500;//temperature_0*std::exp(constant*t);

		if (m_start == 0)
		{
#			pragma omp parallel
			{
				int this_thread = my_omp_get_thread_num(), num_threads = my_omp_get_num_threads();
				size_t chunk_size = (m_particles.size() + num_threads - 1) / num_threads;
				ElementVector::iterator i = m_particles.begin() + this_thread * chunk_size;
				ElementVector::iterator my_end = m_particles.begin() + std::min<size_t>((this_thread + 1)*chunk_size, m_particles.size());
				for (; i != my_end; ++i)
				{
					i->velocity += i->analytical_force * (timestep / 2.0 / Atomic_Masses[i->type]);
					i->position += i->velocity * timestep;

					i->analytical_force = Eigen::Vector3d(0, 0, 0);
					i->numeric_force = Eigen::Vector3d(0, 0, 0);
				}
			}
			if (step % 200 == 0)
			{
				regulateTemperature(temp);
			}
		}

		if (stepCallback)
			stepCallback(step);

		calc2BodyEnergyForces();
		if (calcDistanceCallback)
			calcDistanceCallback();

		calc3BodyEnergyForces();
		if (calcForcesCallback)
			calcForcesCallback();
	}
}

int Crystal_SiC::loadFile(const char* filen)
{
	int num_particles;

	std::ifstream file(filen);
	if (!file) return -1;
	file >> num_particles;
	if (!file) return -1;
	file >> m_length;
	if (!file) return -1;

	m_particles.resize(num_particles);
	m_mass = 0;
	m_end = num_particles;

	for (ElementVector::iterator i = m_particles.begin(); i != m_particles.end(); ++i)
	{
		i->distances.resize(num_particles);
		i->distance_lengths.resize(num_particles);
		i->analytical_force = Eigen::Vector3d(0, 0, 0);
		i->numeric_force = Eigen::Vector3d(0, 0, 0);

		file >> *i;
		if (!file) return -1;
		
		m_mass += Atomic_Masses[i->type];
	}

	return 0;
}

void Crystal_SiC::saveFile(const std::string& filen)
{
	std::ofstream file(filen);
	if (file)
	{
		for (ElementVector::iterator i = m_particles.begin(); i != m_particles.end(); ++i)
			file << *i << std::endl;
	}
	else
	{
		std::cerr << "Error saving file: " << filen << std::endl;
	}
}

std::ostream& operator<< (std::ostream &out, Element_Atom &atom)
{
	switch (atom.type)
	{
	case ATOM_SI:
		out << "Si";
		break;
	case ATOM_C:
		out << "C";
		break;
	default:
		break;
	}
	Eigen::IOFormat format(Eigen::FullPrecision, 0, " ", " ", " ", " ", " ", " ");
	out << " " << atom.position.format(format) << " " << atom.velocity.format(format);
	return out;
}

std::istream& operator>> (std::istream &in, Element_Atom &atom)
{
	std::string type;
	in >> type;
	in >> atom.position(0);
	in >> atom.position(1);
	in >> atom.position(2);

	double vel;
	in >> vel >> vel >> vel;

	if (type == "Si") atom.type = ATOM_SI;
	else if (type == "C") atom.type = ATOM_C;
	return in;
}

