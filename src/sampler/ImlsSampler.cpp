#include "sampler/ImlsSampler.h"

struct by_key{ 
    bool operator()(std::pair<float, int> const &left, std::pair<float, int> const &right) { 
        return left.first < right.first;
    }
};

ImlsSampler::ImlsSampler(int nNumberOfCores, int nSampleNumTHInOneFreedom, float fSamplingSparseSqDisTH):
         mnNumberOfCores(nNumberOfCores), mnSampleNumTHInOneFreedom(nSampleNumTHInOneFreedom), mfSamplingSparseSqDisTH(fSamplingSparseSqDisTH)
{
    mkdtreeFromCorrectPC.reset(new pcl::KdTreeFLANN<PointType>());
}

// Imls平面点采样
pcl::PointCloud<PointType>::Ptr ImlsSampler::SamplingPlanePointsBy9Value(pcl::PointCloud<PointType>::Ptr CorrectPC)
{
	// 使用CorrectPC 构建kd-tree
	mkdtreeFromCorrectPC->setInputCloud(CorrectPC);

	// 并行计算所有点的协方差，从而计算法向量。存储同维度的<法向量，最近点索引和距离>
	std::vector<PointNormalFeature> vPointNormalFeatures = ComputeNormalFeature(CorrectPC);

	// 预设9值向量组（实际上是6组即可）等于点的尺寸，并行遍历点来存储评估指标以及原始点的索引
   	std::vector<std::vector<std::pair<float, int>>> vvEvaluateContainers(6, 
	                      std::vector<std::pair<float, int>>(vPointNormalFeatures.size(), std::make_pair(0.0, -1)));
	

	ComputeEvaluateValue(vPointNormalFeatures, vvEvaluateContainers);

	// 并行排序
#pragma omp parallel for num_threads(mnNumberOfCores) schedule(guided, 8) 
	for(int i=0; i<vvEvaluateContainers.size(); i++)
	{
		std::sort(vvEvaluateContainers[i].begin(), vvEvaluateContainers[i].end(), by_key());
	}

	std::vector<pcl::PointCloud<PointType>::Ptr> vSamplingPCs;
	for(int i=0; i<6; i++)
	{
		pcl::PointCloud<PointType>::Ptr sampling_pc(new pcl::PointCloud<PointType>());
		vSamplingPCs.push_back(sampling_pc);
	}

	// 并行采样子函数
#pragma omp parallel for num_threads(3) schedule(guided, 8) 
	for(int i=0; i<3; i++)
	{
		subRotationSamplingFeatures(vPointNormalFeatures, vvEvaluateContainers[i], vSamplingPCs[i]);
		subTranslationSamplingFeatures(vPointNormalFeatures, vvEvaluateContainers[i+3], vSamplingPCs[i+3]);
	}

	// 合并结果
	pcl::PointCloud<PointType>::Ptr merged_pc(new pcl::PointCloud<PointType>());
	for(int i=0; i<vSamplingPCs.size(); i++)
	{
		*merged_pc += *vSamplingPCs[i];
	}

	return merged_pc;
}

std::vector<PointNormalFeature> ImlsSampler::ComputeNormalFeature(pcl::PointCloud<PointType>::Ptr CorrectPC)
{
    std::vector<PointNormalFeature> vNormalFeatures(CorrectPC->points.size(), PointNormalFeature());
#pragma omp parallel for num_threads(mnNumberOfCores) schedule(guided, 8) 
	for(int i=0; i<CorrectPC->points.size(); i++)
	{
		std::vector<int> pointSearchInd;
		std::vector<float> pointSearchSqDis;
		Eigen::Matrix3d Cov = ComputePointCovByKDtree(CorrectPC, CorrectPC->points[i], pointSearchInd, pointSearchSqDis); // 协方差
		Eigen::Vector3d Normal; double dWeight;
		ComputePointNormalByCov(Cov, Normal, dWeight);
		vNormalFeatures[i] = PointNormalFeature(CorrectPC->points[i], dWeight, Normal, pointSearchInd, pointSearchSqDis);
	}

	return vNormalFeatures;
}

void ImlsSampler::ComputeEvaluateValue(std::vector<PointNormalFeature> vPointNormalFeatures, 
                   std::vector<std::vector<std::pair<float, int>>> &vvEvaluateContainers)
{
    // 预设主轴
	Eigen::Vector3d Xv(1,0,0);
	Eigen::Vector3d Yv(0,1,0);
	Eigen::Vector3d Zv(0,0,1);

#pragma omp parallel for num_threads(mnNumberOfCores) schedule(guided, 8) 
	for(int i=0; i<vPointNormalFeatures.size(); i++)
	{
		PointType point = vPointNormalFeatures[i].mPoint; 
		double a2D2 = vPointNormalFeatures[i].mdWeight * vPointNormalFeatures[i].mdWeight;
		Eigen::Vector3d p(point.x, point.y, point.z);
		Eigen::Vector3d pCrossn = skew_symmetric(p) * vPointNormalFeatures[i].mNormal;

		float k1 = a2D2 * pCrossn.transpose() * Xv; // 角度
		float k2 = a2D2 * pCrossn.transpose() * Yv;
		float k3 = a2D2 * pCrossn.transpose() * Zv;
		float k4 = a2D2 * std::fabs(vPointNormalFeatures[i].mNormal.transpose()*Xv);  // 位移
		float k5 = a2D2 * std::fabs(vPointNormalFeatures[i].mNormal.transpose()*Yv);
		float k6 = a2D2 * std::fabs(vPointNormalFeatures[i].mNormal.transpose()*Zv);

		vvEvaluateContainers[0][i] = std::make_pair(k1, i);
		vvEvaluateContainers[1][i] = std::make_pair(k2, i);
		vvEvaluateContainers[2][i] = std::make_pair(k3, i);
		vvEvaluateContainers[3][i] = std::make_pair(k4, i);
		vvEvaluateContainers[4][i] = std::make_pair(k5, i);
		vvEvaluateContainers[5][i] = std::make_pair(k6, i);
	}
}

void ImlsSampler::subRotationSamplingFeatures(std::vector<PointNormalFeature> vPointNormalFeatures,
             std::vector<std::pair<float, int>> ordered_vector, pcl::PointCloud<PointType>::Ptr samplingPC)
{
    int niter = 0;
	int cur_select_point_num = 0;

	// 等于所有点的维度的true向量EfficientPlanePointflags。
	std::vector<bool> vEfficientPlanePointflags(vPointNormalFeatures.size(), true);
	for(int i=ordered_vector.size()-1; i>=0; i--)
	{
		niter++;
		// 检查当前点是否因为之前已提取的点而屏蔽
		if(vEfficientPlanePointflags[i])  // 对于一个新的点，直接存储到当前自由度的结果点云，有原始id号码
		{
			vEfficientPlanePointflags[i] = false;

			// 因为我们的框架后端使用的是平面路标，因此，测量也进行额外的平面权重判定
			if(vPointNormalFeatures[ordered_vector[i].second].mdWeight<0.5)
				continue;

			samplingPC->push_back(vPointNormalFeatures[ordered_vector[i].second].mPoint);
	
			// 找到最近点集合EfficientPlanePointflags设置为false，并将距离足够近的屏蔽掉
			for(int j=0; j<vPointNormalFeatures[i].mPointSearchSqDis.size(); j++)
			{
			 	if(vPointNormalFeatures[i].mPointSearchSqDis[j] < mfSamplingSparseSqDisTH)
					vEfficientPlanePointflags[vPointNormalFeatures[i].mPointSearchInd[j]] = false;
			}

			cur_select_point_num++;
			if(cur_select_point_num>=mnSampleNumTHInOneFreedom)
				break;
		}

		if(niter>=3*mnSampleNumTHInOneFreedom)
			break;
	}

	niter = 0;
	cur_select_point_num = 0;
	vEfficientPlanePointflags = std::vector<bool>(vPointNormalFeatures.size(), true);
	for(int i=0; i<ordered_vector.size(); i++)
	{
		niter++;
		if(vEfficientPlanePointflags[i])
		{
			vEfficientPlanePointflags[i] = false;

			// 因为我们的框架后端使用的是平面路标，因此，测量也进行额外的平面权重判定
			if(vPointNormalFeatures[ordered_vector[i].second].mdWeight<0.5)
				continue;

			samplingPC->push_back(vPointNormalFeatures[ordered_vector[i].second].mPoint);
			
			// 找到最近点集合EfficientPlanePointflags设置为false，并将距离足够近的屏蔽掉
			for(int j=0; j<vPointNormalFeatures[i].mPointSearchSqDis.size(); j++)
			{
			 	if(vPointNormalFeatures[i].mPointSearchSqDis[j] < mfSamplingSparseSqDisTH)
					vEfficientPlanePointflags[vPointNormalFeatures[i].mPointSearchInd[j]] = false;
			}

			cur_select_point_num++;
			if(cur_select_point_num>=mnSampleNumTHInOneFreedom)
				break;
		}

		if(niter>=3*mnSampleNumTHInOneFreedom)
			break;
	}
}

void ImlsSampler::subTranslationSamplingFeatures(std::vector<PointNormalFeature> vPointNormalFeatures, std::vector<std::pair<float, int>> ordered_vector,
                           pcl::PointCloud<PointType>::Ptr samplingPC)
{
    int niter = 0;
	int cur_select_point_num = 0;
	std::vector<bool> vEfficientPlanePointflags(vPointNormalFeatures.size(), true);
	for(int i=ordered_vector.size()-1; i>=0; i--)
	{
		niter++;
		// 检查当前点是否因为之前已提取的点而屏蔽
		if(vEfficientPlanePointflags[i])  // 对于一个新的点，直接存储到当前自由度的结果点云，有原始id号码
		{
			vEfficientPlanePointflags[i] = false;

			// 因为我们的框架后端使用的是平面路标，因此，测量也进行额外的平面权重判定
			if(vPointNormalFeatures[ordered_vector[i].second].mdWeight<0.6)
				continue;

			samplingPC->push_back(vPointNormalFeatures[ordered_vector[i].second].mPoint);
	
			// 找到最近点集合EfficientPlanePointflags设置为false，并将距离足够近的屏蔽掉
			for(int j=0; j<vPointNormalFeatures[i].mPointSearchSqDis.size(); j++)
			{
			 	if(vPointNormalFeatures[i].mPointSearchSqDis[j] < mfSamplingSparseSqDisTH)
				{
					vEfficientPlanePointflags[vPointNormalFeatures[i].mPointSearchInd[j]] = false;
				}	
			}

			cur_select_point_num++;
			if(cur_select_point_num>=mnSampleNumTHInOneFreedom)
				break;
		}

		if(niter>=3*mnSampleNumTHInOneFreedom)
			break;
	}
}

Eigen::Matrix3d ImlsSampler::ComputePointCovByKDtree(pcl::PointCloud<PointType>::Ptr CorrectPC, PointType point,
                std::vector<int> pointSearchInd, std::vector<float> pointSearchSqDis)
{
    mkdtreeFromCorrectPC->nearestKSearch(point, 10, pointSearchInd, pointSearchSqDis);

	Eigen::MatrixXd MeaMatrix(pointSearchInd.size(), 3);
	for(int i=0; i<pointSearchInd.size(); i++)
	{
		PointType nearest_point = CorrectPC->points[pointSearchInd[i]];
		MeaMatrix(i, 0) = nearest_point.x;
		MeaMatrix(i, 1) = nearest_point.y;
		MeaMatrix(i, 2) = nearest_point.z;
	}
	Eigen::Matrix<double, 1, 3> Center = MeaMatrix.colwise().mean().eval();
	MeaMatrix.rowwise() -= Center; 

	Eigen::Matrix3d Cov = MeaMatrix.transpose() * MeaMatrix / (MeaMatrix.rows() - 1);

	// Eigen::Matrix4d Cov_mea; Cov_mea.setZero();
	// Cov_mea.block(0,0,3,3) = Cov; 

	return Cov;
}

void ImlsSampler::ComputePointNormalByCov(Eigen::Matrix3d Cov, Eigen::Vector3d &Normal, double &dWeight)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;

	es.compute(Cov);
	double min_eigenval = 1e10;
	int min_i; 
	for (int i = 0; i < 3; i++) {
		double curr_eigen = fabs(es.eigenvalues()(i));
		if (curr_eigen < min_eigenval) {
			min_eigenval = curr_eigen;
			min_i = i;
		}
	} 

	Normal = es.eigenvectors().col(min_i);
	Normal = Normal / Normal.norm();

	Eigen::Vector3d ev = es.eigenvalues();

	if(ev.z()>0)
	{
		dWeight = (std::sqrt(std::fabs(ev.y()))-std::sqrt(std::fabs(ev.x()))) / std::sqrt(ev.z());
	}
	else
		dWeight = 0;
}