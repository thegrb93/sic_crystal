#ifdef ENABLE_MPI
#include "crystal.h"
#include <mpi.h>
#include <fstream>

Crystal_SiC crystal;
int mpi_err, numnodes, rank, num_particles, mpi_chunk_size;
MPI_Datatype mpi_posvel;
MPI_Datatype mpi_types;
MPI_Datatype mpi_forces;

void distancesFinished()
{
	//Broadcast all calculated distances between the nodes
	for (int i = crystal.m_start; i < crystal.m_end; ++i)
	{
		mpi_err = MPI_Bcast(crystal.m_particles[i].distances.data(), sizeof(Eigen::Vector3d)*num_particles, MPI_BYTE, rank, MPI_COMM_WORLD);
		mpi_err = MPI_Bcast(crystal.m_particles[i].distance_lengths.data(), num_particles, MPI_DOUBLE, rank, MPI_COMM_WORLD);
	}
	//Receive distances from other nodes
	for (int i = 0; i < numnodes; ++i)
	{
		if (i != rank)
		{
			int mpi_chunk_start = i*mpi_chunk_size;
			int mpi_chunk_end = std::min<int>(mpi_chunk_start + mpi_chunk_size, num_particles);
			for (int j = mpi_chunk_start; j < mpi_chunk_end; ++j)
			{
				mpi_err = MPI_Bcast(crystal.m_particles[j].distances.data(), sizeof(Eigen::Vector3d)*num_particles, MPI_BYTE, i, MPI_COMM_WORLD);
				mpi_err = MPI_Bcast(crystal.m_particles[j].distance_lengths.data(), num_particles, MPI_DOUBLE, i, MPI_COMM_WORLD);
			}
		}
	}
}

void forcesFinished()
{
	//Send all calculated forces to the root node
	if (rank != 0)
	{
		mpi_err = MPI_Send(crystal.m_particles.data(), num_particles, mpi_forces, 0, 0, MPI_COMM_WORLD);
	}
	//Receive forces from other nodes
	if (rank == 0)
	{
		for (int i = 1; i < numnodes; ++i)
		{
			Eigen3dVector forces(num_particles);
			mpi_err = MPI_Recv(forces.data(), sizeof(Eigen::Vector3d)*num_particles, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#			pragma omp parallel for
			for (int j = 0; j < num_particles; ++j)
			{
				crystal.m_particles[j].analytical_force += forces[j];
			}
		}
	}

	/*std::ofstream forces("forces_mpi.txt");
	Eigen::IOFormat format(Eigen::FullPrecision, 0, " ", " ", " ", " ", " ", " ");
	for (ElementVector::iterator i = crystal.m_particles.begin(); i != crystal.m_particles.end(); ++i)
		forces << i->analytical_force.format(format) << std::endl;*/
}

//Update positions and velocities to the nodes
void stepFinished(int step)
{
	mpi_err = MPI_Bcast(crystal.m_particles.data(), num_particles, mpi_posvel, 0, MPI_COMM_WORLD);

	if (rank == 0 && step % 1000 == 0)
	{
		std::stringstream filen;
		filen << "data/crystal_mpi_" << step/1000 << ".xyz";
		crystal.saveFile(filen.str());
	}
}

int main(int argc, char **argv)
{
	mpi_err = MPI_Init(&argc, &argv);
	if (mpi_err) return -1;
	mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &numnodes);
	if (mpi_err) return -1;
	mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (mpi_err) return -1;

	//Set up mpi data structures
	{
		int mpi_types_lens[] = { 1 };
		int mpi_posvel_lens[] = { sizeof(Eigen::Vector3d), sizeof(Eigen::Vector3d), 1 };
		int mpi_force_lens[] = { sizeof(Eigen::Vector3d), 1 };
		MPI_Aint mpi_types_offsets[] = { offsetof(Element_Atom, type), sizeof(Element_Atom) };
		MPI_Aint mpi_posvel_offsets[] = { offsetof(Element_Atom, type), offsetof(Element_Atom, position), offsetof(Element_Atom, velocity), sizeof(Element_Atom) };
		MPI_Aint mpi_force_offsets[] = { offsetof(Element_Atom, analytical_force), sizeof(Element_Atom) };
		int mpi_types_types[] = { MPI_INT32_T, MPI_UB };
		int mpi_posvel_types[] = { MPI_BYTE, MPI_BYTE, MPI_UB };
		int mpi_force_types[] = { MPI_BYTE, MPI_UB };

		mpi_err = MPI_Type_create_struct(2, mpi_types_lens, mpi_types_offsets, mpi_types_types, &mpi_types);
		mpi_err = MPI_Type_create_struct(3, mpi_posvel_lens, mpi_posvel_offsets, mpi_posvel_types, &mpi_posvel);
		mpi_err = MPI_Type_create_struct(2, mpi_force_lens, mpi_force_offsets, mpi_force_types, &mpi_forces);

		MPI_Type_commit(&mpi_types);
		MPI_Type_commit(&mpi_posvel);
		MPI_Type_commit(&mpi_forces);
	}

	crystal.calcDistanceCallback = &distancesFinished;
	crystal.calcForcesCallback = &forcesFinished;
	crystal.stepCallback = &stepFinished;

	if (rank == 0)
	{
		if (argc < 2)
		{
			std::cout << "Expected an input crystal text file..." << std::endl;
			return 0;
		}
		if (crystal.loadFile(argv[1]))
		{
			std::cout << "Error reading crystal file." << std::endl;
			return 0;
		}
		num_particles = crystal.m_particles.size();

		const double temperature_0 = 3500;
		crystal.randomizeVelocity(temperature_0);
	}

	mpi_err = MPI_Bcast(&num_particles, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
	mpi_err = MPI_Bcast(&crystal.m_length, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	mpi_err = MPI_Bcast(&crystal.m_mass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank != 0)
	{
		crystal.m_particles.resize(num_particles);
#		pragma omp parallel for
		for (int i = 0; i < num_particles; ++i)
		{
			Element_Atom& atom = crystal.m_particles[i];
			atom.distances.resize(num_particles);
			atom.distance_lengths.resize(num_particles);
		}
	}

	mpi_err = MPI_Bcast(crystal.m_particles.data(), num_particles, mpi_types, 0, MPI_COMM_WORLD);

	mpi_chunk_size = (num_particles + numnodes - 1) / numnodes;
	crystal.m_start = rank * mpi_chunk_size;
	crystal.m_end = std::min<size_t>(crystal.m_start + mpi_chunk_size, num_particles);

	crystal.simulate(1, 2000000); //2 million steps 1fm each

	MPI_Finalize();
}
#endif