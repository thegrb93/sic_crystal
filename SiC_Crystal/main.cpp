#ifndef ENABLE_MPI
#include "crystal.h"
#include <fstream>
Crystal_SiC crystal;

//#define Visualization
#ifdef Visualization
#include <pcl/visualization/cloud_viewer.h>
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;
pcl::visualization::CloudViewer* pviewer = nullptr;
#endif


void stepcallback(int step)
{
	#ifdef Visualization
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>(crystal.m_particles.size(),1));
		clouds.push_back(cloud);
		for (size_t i = 0; i<cloud->size(); ++i)
		{
			pcl::PointXYZRGB& point = cloud->operator[](i);
			Element_Atom& atom = crystal.m_particles[i];
		
			point.x = atom.position[0];
			point.y = atom.position[1];
			point.z = atom.position[2];
			point.r = atom.type==ATOM_SI ? 255 : 0;
			point.g = 0;
			point.b = atom.type==ATOM_C ? 255 : 0;
		}
	
		if(pviewer)
		{
			pviewer->showCloud(cloud);
			pviewer->runOnVisualizationThreadOnce([](pcl::visualization::PCLVisualizer& vis)
			{
				vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5);
			});
		}
	#endif

	if (step % 1000 == 0)
	{
		std::cout << "Iteration #: " << step << std::endl;
		std::stringstream filen;
		filen << "data/crystal_" << step << ".xyz";
		crystal.saveFile(filen.str());
	}
}

void forcecallback()
{
	std::ofstream forces("forces_omp.txt");
	Eigen::IOFormat format(Eigen::FullPrecision, 0, " ", " ", " ", " ", " ", " ");
	for (ElementVector::iterator i = crystal.m_particles.begin(); i != crystal.m_particles.end(); ++i)
		forces << i->analytical_force.format(format) << std::endl;
}

int main(int argc, char **argv)
{
	if(argc < 2)
	{
		std::cout << "Expected an input crystal text file..." << std::endl;
		return 0;
	}
	
	if(crystal.loadFile(argv[1]))
	{
		std::cout << "Error reading crystal file." << std::endl;
		return 0;
	}
	
	#ifdef Visualization
		pcl::visualization::CloudViewer viewer ("Atoms");
		pviewer = &viewer;
	#endif

	crystal.stepCallback = &stepcallback;
	//crystal.calcForcesCallback = &forcecallback;

	const double temperature_0 = 3500;
	crystal.randomizeVelocity(temperature_0);

	crystal.simulate(1, 2000000);
	
	return 0;
}
#endif