# m-imls_sampler

**Implementation of the m-imls_sampler as described in the paper "IMLS-SLAM: scan-to-model matching based on 3D data", of Deschaud et al. (ICRA 2018).**

## 0. Features
Extract a set of structural plane points from range image using region growing algorithm. In order to address common occlusion problem, this algorithm uses a planar set based on the closest point representation to construct a kd-tree, and merges adjacent planar point sets that are less than the distance threshold.

## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
**Ubuntu >= 18.04**

ROS    >= Melodic. [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 1.2. **PCL && Eigen**
PCL    >= 1.8,   Follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html).

Eigen  >= 3.3.3, Follow [Eigen Installation](http://eigen.tuxfamily.org/index.php?title=Main_Page).

## 2. Build

Clone the repository and catkin_make:

```
    cd ~/$A_ROS_DIR$/src
    git clone https://github.com/Hero941215/plane_detector_cpm
    cd plane_detector_cpm
    mkdir build
    cd build
    cmake ..
    make -j8
```

## 3. Run
### 3.1. **run demo**

    ./test_plane_detector_scan_merge2 /your_pcd_path

## 4. Acknowledgments

Thanks for [plane_detector](https://github.com/robotics-upo/plane_detector).

