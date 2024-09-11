#ifndef IMLS_SAMPLER_H
#define IMLS_SAMPLER_H

#include <omp.h> 

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <eigen3/Eigen/Dense>

#include <vector>
#include <string>
#include <list>
#include <mutex>
#include <memory>
#include <thread>
#include <unistd.h>
#include <iomanip>

typedef pcl::PointXYZI PointType;

template <typename T>
Eigen::Matrix<T, 3, 3> skew_symmetric(Eigen::Matrix<T, 3, 1> v) {
  Eigen::Matrix<T, 3, 3> S;
  S << T(0), -v(2), v(1), v(2), T(0), -v(0), -v(1), v(0), T(0);
  return S;
}

class PointNormalFeature
{
public:
	PointNormalFeature(PointType point, double dWeight, Eigen::Vector3d Normal, std::vector<int> PointSearchInd, std::vector<float> PointSearchSqDis): 
	           mPoint(point), mdWeight(dWeight), mNormal(Normal), mPointSearchInd(PointSearchInd),  mPointSearchSqDis(PointSearchSqDis)
	{}

	PointNormalFeature(): mdWeight(0)
	{
		mNormal.setZero();
	}

public:
	PointType mPoint;
	double mdWeight;
	Eigen::Vector3d mNormal;
	std::vector<int> mPointSearchInd;
	std::vector<float> mPointSearchSqDis;
};

class ImlsSampler
{
public:
    ImlsSampler(int nNumberOfCores, int nSampleNumTHInOneFreedom, float fSamplingSparseSqDisTH);

	// Imls平面点采样
	pcl::PointCloud<PointType>::Ptr SamplingPlanePointsBy9Value(pcl::PointCloud<PointType>::Ptr CorrectPC);

protected:
	std::vector<PointNormalFeature> ComputeNormalFeature(pcl::PointCloud<PointType>::Ptr CorrectPC);
	void ComputeEvaluateValue(std::vector<PointNormalFeature> vPointNormalFeatures, std::vector<std::vector<std::pair<float, int>>> &vvEvaluateContainers);
	void subRotationSamplingFeatures(std::vector<PointNormalFeature> vPointNormalFeatures, std::vector<std::pair<float, int>> ordered_vector, pcl::PointCloud<PointType>::Ptr samplingPC);
	void subTranslationSamplingFeatures(std::vector<PointNormalFeature> vPointNormalFeatures, std::vector<std::pair<float, int>> ordered_vector,
	                      pcl::PointCloud<PointType>::Ptr samplingPC);
	Eigen::Matrix3d ComputePointCovByKDtree(pcl::PointCloud<PointType>::Ptr CorrectPC, PointType point, std::vector<int> pointSearchInd, std::vector<float> pointSearchSqDis);
	void ComputePointNormalByCov(Eigen::Matrix3d Cov, Eigen::Vector3d &Normal, double &dWeight);    

protected:

	// imls 采样的相关数据结构和参数
    int mnNumberOfCores;
	int mnSampleNumTHInOneFreedom;  // 200
	float mfSamplingSparseSqDisTH;   // 0.05

    pcl::KdTreeFLANN<PointType>::Ptr mkdtreeFromCorrectPC; 
};

#endif // IMLS_SAMPLER_H
