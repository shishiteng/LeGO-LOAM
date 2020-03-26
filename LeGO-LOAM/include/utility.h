#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#include "cloud_msgs/cloud_info.h"

#include <opencv/cv.h>

#include <Eigen/Eigen>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#define PI 3.14159265

using namespace std;

typedef pcl::PointXYZI PointType;

extern const string pointCloudTopic = "/rslidar_points";
extern const string imuTopic = "/imu/data";

// Save pcd
extern const string fileDirectory = "/tmp/";

// VLP-16
extern const int N_SCAN = 16;
extern const int Horizon_SCAN = 1800;
extern const float ang_res_x = 0.2;
extern const float ang_res_y = 2.0;
extern const float ang_bottom = 15.0 + 0.1;
extern const int groundScanInd = 7;

// HDL-32E
// extern const int N_SCAN = 32;
// extern const int Horizon_SCAN = 1800;
// extern const float ang_res_x = 360.0/float(Horizon_SCAN);
// extern const float ang_res_y = 41.33/float(N_SCAN-1);
// extern const float ang_bottom = 30.67;
// extern const int groundScanInd = 20;

// Ouster users may need to uncomment line 159 in imageProjection.cpp
// Usage of Ouster imu data is not fully supported yet, please just publish point cloud data
// Ouster OS1-16
// extern const int N_SCAN = 16;
// extern const int Horizon_SCAN = 1024;
// extern const float ang_res_x = 360.0/float(Horizon_SCAN);
// extern const float ang_res_y = 33.2/float(N_SCAN-1);
// extern const float ang_bottom = 16.6+0.1;
// extern const int groundScanInd = 7;

// Ouster OS1-64
// extern const int N_SCAN = 64;
// extern const int Horizon_SCAN = 1024;
// extern const float ang_res_x = 360.0/float(Horizon_SCAN);
// extern const float ang_res_y = 33.2/float(N_SCAN-1);
// extern const float ang_bottom = 16.6+0.1;
// extern const int groundScanInd = 15;

extern const bool loopClosureEnableFlag = true;
extern const double mappingProcessInterval = 0.3;

extern const float scanPeriod = 0.1;
extern const int systemDelay = 0;
extern const int imuQueLength = 200;

extern const float sensorMountAngle = 0.0;
extern const float segmentTheta = 60.0 / 180.0 * M_PI; // decrese this value may improve accuracy
extern const int segmentValidPointNum = 5;
extern const int segmentValidLineNum = 3;
extern const float segmentAlphaX = ang_res_x / 180.0 * M_PI;
extern const float segmentAlphaY = ang_res_y / 180.0 * M_PI;

extern const int edgeFeatureNum = 2;
extern const int surfFeatureNum = 4;
extern const int sectionsTotal = 6;
extern const float edgeThreshold = 0.1;
extern const float surfThreshold = 0.1;
extern const float nearestFeatureSearchSqDist = 25;

// Mapping Params
extern const float surroundingKeyframeSearchRadius = 50.0; // key frame that is within n meters from current pose will be considerd for scan-to-map optimization (when loop closure disabled)
extern const int surroundingKeyframeSearchNum = 50;        // submap size (when loop closure enabled)
// history key frames (history submap for loop closure)
extern const float historyKeyframeSearchRadius = 7.0; // key frame that is within n meters from current pose will be considerd for loop closure
extern const int historyKeyframeSearchNum = 25;       // 2n+1 number of hostory key frames will be fused into a submap for loop closure
extern const float historyKeyframeFitnessScore = 0.3; // the smaller the better alignment

extern const float globalMapVisualizationSearchRadius = 500.0; // key frames with in n meters will be visualized

struct smoothness_t
{
    float value;
    size_t ind;
};

struct by_value
{
    bool operator()(smoothness_t const &left, smoothness_t const &right)
    {
        return left.value < right.value;
    }
};

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))

typedef PointXYZIRPYT PointTypePose;

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3> SkewSymmetric(const Eigen::MatrixBase<Derived> &v3d)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 3> m;
    m << typename Derived::Scalar(0), -v3d.z(), v3d.y(),
        v3d.z(), typename Derived::Scalar(0), -v3d.x(),
        -v3d.y(), v3d.x(), typename Derived::Scalar(0);
    return m;
}

template<typename Derived>
inline Eigen::Quaternion<typename Derived::Scalar> DeltaQ(const Eigen::MatrixBase<Derived> &theta) {
  typedef typename Derived::Scalar Scalar_t;

  Eigen::Quaternion<Scalar_t> dq;
  Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
  half_theta /= static_cast<Scalar_t>(2.0);
  dq.w() = static_cast<Scalar_t>(1.0);
  dq.x() = half_theta.x();
  dq.y() = half_theta.y();
  dq.z() = half_theta.z();
  return dq;
}

bool isinvalid(PointType p)
{
    return (isnan(p.x) || isinf(p.x) ||
            isnan(p.y) || isinf(p.y) ||
            isnan(p.z) || isinf(p.z));
}

void Vector2Eigen(float *v, Eigen::Matrix3f &rot, Eigen::Vector3f &pos)
{
    tf::Matrix3x3 m;
    m.setRPY((double)v[0], (double)v[1], (double)v[2]);
    rot << m[0][0], m[0][1], m[0][2],
        m[1][0], m[1][1], m[1][2],
        m[2][0], m[2][1], m[2][2];

    pos = Eigen::Vector3f(v[3], v[4], v[5]);
}

void Vector2Eigen(float *v, Eigen::Matrix4f &trans)
{
    Eigen::Matrix3f rot;
    Eigen::Vector3f pos;
    Vector2Eigen(v, rot, pos);

    trans = Eigen::Matrix4f::Identity();
    trans.block(0, 0, 3, 3) = rot;
    trans.block(0, 3, 3, 1) = pos;
}

void Eigen2Vector(Eigen::Matrix4f trans, float *v)
{
    tf::Matrix3x3 m((double)trans(0, 0), (double)trans(0, 1), (double)trans(0, 2),
                    (double)trans(1, 0), (double)trans(1, 1), (double)trans(1, 2),
                    (double)trans(2, 0), (double)trans(2, 1), (double)trans(2, 2));
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    v[0] = (float)roll;
    v[1] = (float)pitch;
    v[2] = (float)yaw;
    v[3] = trans(0, 3);
    v[4] = trans(1, 3);
    v[5] = trans(2, 3);
}

void Eigen2Vector(Eigen::Matrix3f rot, Eigen::Vector3f pos, float *v)
{
    tf::Matrix3x3 m((double)rot(0, 0), (double)rot(0, 1), (double)rot(0, 2),
                    (double)rot(1, 0), (double)rot(1, 1), (double)rot(1, 2),
                    (double)rot(2, 0), (double)rot(2, 1), (double)rot(2, 2));
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    v[0] = (float)roll;
    v[1] = (float)pitch;
    v[2] = (float)yaw;
    v[3] = pos[0];
    v[4] = pos[1];
    v[5] = pos[2];
}

#endif
