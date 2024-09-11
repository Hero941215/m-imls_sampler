


#include <thread>

#include <vector>
#include <random>
#include <queue>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Eigenvalues>
// #include <boost/array.hpp>
// #include <boost/thread.hpp>
#include <exception>

#include <iostream>
#include <algorithm>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "tic_toc.h"

#include "sampler/ImlsSampler.h"

#include "tsl/robin_map.h"


using namespace std;

typedef pcl::PointXYZI PointType;

class Vector3iHash {
public:
	size_t operator()(const Eigen::Vector3i& x) const {
		size_t seed = 0;
		boost::hash_combine(seed, x[0]);
		boost::hash_combine(seed, x[1]);
		boost::hash_combine(seed, x[2]);
		return seed;
	}
};

using GridVoxelSampler = tsl::robin_map<Eigen::Vector3i, PointType, Vector3iHash, std::equal_to<Eigen::Vector3i>, 
Eigen::aligned_allocator<std::pair<const Eigen::Vector3i, PointType>>>;

const double dssize = 0.2;

Eigen::Vector3i ComputeVoxelCoord(Eigen::Vector3d pw)
{
    double loc_xyz[3];
	for(int j=0; j<3; j++)
	{
		loc_xyz[j] = pw[j] / dssize;
		if(loc_xyz[j] < 0)
		{
			loc_xyz[j] -= 1.0;
		}
	}

	Eigen::Vector3i VoxelCoord((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
	return VoxelCoord;
}

void pc_show(pcl::PointCloud<PointType>::Ptr sampling_pc);


int main(int argc, char **argv) 
{
    if (argc < 2) {
        printf("Usage: %s <point cloud> [<delta> <epsilon> <gamma> <theta>]\n", argv[0]);
        return -1;
    }

    // 载入点云
    const std::string pcd_file(argv[1]);
    pcl::PointCloud<PointType>::Ptr point_cloud(
        new pcl::PointCloud<PointType>);

    if (pcl::io::loadPCDFile<PointType>(pcd_file, *point_cloud) == -1) {
        std::cout << "Couldn't read pcd file!\n";
        return -1;
    }

    TicToc t_whole;
    //  pcl::PointCloud<PointType>::Ptr point_cloud2(
    //     new pcl::PointCloud<PointType>);
    // pcl::VoxelGrid<PointType> mDownSizeFilter;
    // mDownSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    // mDownSizeFilter.setInputCloud(point_cloud);
	// mDownSizeFilter.filter(*point_cloud2); // 进行帧首点云降采样

    // 将点遍历输入，保留体素中的唯一点
    pcl::PointCloud<PointType>::Ptr sampling_pc(
        new pcl::PointCloud<PointType>);

    GridVoxelSampler sampler;
    PointType point;
    for(int i=0; i<point_cloud->points.size(); i++)
    {
        Eigen::Vector3d p3d(point_cloud->points[i].x, point_cloud->points[i].y, point_cloud->points[i].z);

        // 计算体素坐标
        Eigen::Vector3i vc = ComputeVoxelCoord(p3d);
        if (sampler.find(vc) == sampler.end()) {
                sampler[vc] = point_cloud->points[i];
        }
    }

    for (auto it = sampler.begin(); it != sampler.end(); ++it) 
		sampling_pc->push_back(it->second);

    TicToc t_ilms;
    // 将点遍历输入，保留体素中的唯一点
    ImlsSampler sampler2(8, 100, 0.05);
    pcl::PointCloud<PointType>::Ptr sampling_pc2 = sampler2.SamplingPlanePointsBy9Value(sampling_pc);
    printf("imls sampling time %f ms \n \n", t_ilms.toc());

    printf("whole sampling time %f ms \n \n", t_whole.toc());

    std::cout << "point_>points.size(): " << sampling_pc2->points.size() << std::endl;

    pc_show(sampling_pc2);

    return 0;
}

void pc_show(pcl::PointCloud<PointType>::Ptr sampling_pc)
{
    int r = 255;
    int g = 0;
    int b = 0;

    // 显示容器
    pcl::PointXYZRGB rgb_point;
    pcl::PointXYZI thisPoint;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_pointcloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

    for(int i=0; i<sampling_pc->points.size(); i++)
    {
        thisPoint = sampling_pc->points[i];
        rgb_point.x = thisPoint.x;
        rgb_point.y = thisPoint.y;
        rgb_point.z = thisPoint.z;
        rgb_point.r = r;
        rgb_point.g = g;
        rgb_point.b = b; 
        rgb_pointcloud->push_back(rgb_point);
    }

    // pcl 显示所有检测到的平面
    pcl::visualization::PCLVisualizer::Ptr visualizer(
        new pcl::visualization::PCLVisualizer("Sampling Visualizer"));
    visualizer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>
        rgb_color_handler(rgb_pointcloud);
    visualizer->addPointCloud<pcl::PointXYZRGB>(rgb_pointcloud, rgb_color_handler,
                                                "RGB PointCloud");
    visualizer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "RGB PointCloud");
    visualizer->addCoordinateSystem(5.0);                                                                  

    while (!visualizer->wasStopped()) {
        visualizer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}