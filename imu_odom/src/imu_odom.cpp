#include <stdio.h>
#include <math.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>

#include "Eigen/Eigen"

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

nav_msgs::Path path_;

using namespace Eigen;
using namespace std;

#define USE_IMU_POSE 0

/**************预积分***********************/
class IntegrationOdometry
{
public:
    Eigen::Vector3d vel_0, vel_1;
    Eigen::Vector3d gyr_0, gyr_1;
    double dt;
    double sum_dt;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;

public:
    // IntegrationOdometry() = delete;
    IntegrationOdometry()
    {
        vel_0 = Eigen::Vector3d::Zero();
        vel_1 = Eigen::Vector3d::Zero();
        gyr_0 = Eigen::Vector3d::Zero();
        gyr_1 = Eigen::Vector3d::Zero();
        delta_p = Eigen::Vector3d::Zero();
        delta_v = Eigen::Vector3d::Zero();
        delta_q = Eigen::Quaterniond::Identity();
    }

    IntegrationOdometry(const Eigen::Vector3d &_vel_0, const Eigen::Vector3d &_gyr_0)
        : vel_0{_vel_0}, gyr_0{_gyr_0}, sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()},
          delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()}

    {
    }

    void midPointIntegration(double _dt,
                             const Eigen::Vector3d &_vel_0, const Eigen::Vector3d &_gyr_0,
                             const Eigen::Vector3d &_vel_1, const Eigen::Vector3d &_gyr_1,
                             const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                             Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v)
    {
        //ROS_INFO("midpoint integration");
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1);
        result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        Vector3d un_vel_0 = delta_q * (_vel_0);
        Vector3d un_vel_1 = result_delta_q * (_vel_1);
        Vector3d un_vel = 0.5 * (un_vel_0 + un_vel_1);
        //cout<< "v:"<<delta_v.transpose()<<endl;
        //cout<< "p:"<<delta_p.transpose()<<endl;
        //cout<< "dt:"<<_dt<<endl;
        result_delta_p = delta_p + delta_v * _dt;
        //cout<< "dp:"<<result_delta_p.transpose()<<endl;
        //result_delta_v = delta_v;
        result_delta_v = un_vel;
    }

    void propagate(double _dt, const Eigen::Vector3d &_vel_1, const Eigen::Vector3d &_gyr_1)
    {
        dt = _dt;
        vel_1 = _vel_1;
        gyr_1 = _gyr_1;
        Vector3d result_delta_p(Eigen::Vector3d::Zero());
        Vector3d result_delta_v(Eigen::Vector3d::Zero());
        Quaterniond result_delta_q;
#if 0
    printf("propagete_input: t[%.3f]\n v0[%.3f %.3f %.3f] w[%.3f %.3f %.3f]\n v1[%.3f %.3f %.3f] w[%.3f %.3f %.3f]\n",
	   dt,
	   vel_0[0],vel_0[1],vel_0[2],
	   gyr_0[0],gyr_0[1],gyr_0[2],
	   vel_1[0],vel_1[1],vel_1[2],
	   gyr_1[0],gyr_1[1],gyr_1[2]);
#endif

        //
        midPointIntegration(_dt, vel_0, gyr_0, _vel_1, _gyr_1, delta_p, delta_q, delta_v,
                            result_delta_p, result_delta_q, result_delta_v);

        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        delta_q.normalize();
        sum_dt += dt;
        vel_0 = vel_1;
        gyr_0 = gyr_1;

        //    printf("propagete_output: delta_p[%.3f %.3f %.3f]\n",
        //	   result_delta_p[0],result_delta_p[1],result_delta_p[2]);
    }
};
/*********************************/

ros::Publisher pub_odom_;
ros::Publisher pub_path_;
double last_time_ = 0;
Eigen::Vector3d last_eular_(0, 0, 0);
tf::Quaternion last_q_;
IntegrationOdometry *pintegration_ = NULL;

Eigen::Matrix3d i2o_;

void callback(const nav_msgs::OdometryConstPtr &odom_msg, const sensor_msgs::ImuConstPtr &imu_msg, const sensor_msgs::PointCloud2ConstPtr &points_msg)
{
    ROS_DEBUG("callback: odom_%lf imu_%lf points_%lf", odom_msg->header.stamp.toSec(), imu_msg->header.stamp.toSec(), points_msg->header.stamp.toSec());

#if USE_IMU_POSE
    double roll, pitch, yaw;
    tf::Quaternion q;
    tf::quaternionMsgToTF(imu_msg->orientation, q);
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
    Eigen::Vector3d eular_angles(roll, pitch, yaw);
#endif

    if (!pintegration_)
    {
        pintegration_ = new IntegrationOdometry();
        last_time_ = imu_msg->header.stamp.toSec();
#if USE_IMU_POSE
        last_eular_ = eular_angles;
        last_q_ = q;
#endif
    }
    else
    {
        double dt = imu_msg->header.stamp.toSec() - last_time_;
        Eigen::Vector3d linear_velocity(odom_msg->twist.twist.linear.x, odom_msg->twist.twist.linear.y, odom_msg->twist.twist.linear.z);
        Eigen::Vector3d angular_velocity(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);

#if USE_IMU_POSE
        // 带姿态角的IMU用姿态变化算出角速度，不带姿态角的IMU直接使用角速度
        // 不能直接用欧拉角相减，正负180度相减会有跳变
        Eigen::Vector3d deular;
        tf::Quaternion dq = last_q_.inverse() * q;
        tf::Matrix3x3(dq).getRPY(deular[0], deular[1], deular[2]);
        angular_velocity = deular / dt;
#endif

        // 把角速度从imu坐标系转到odom坐标系下
        angular_velocity = i2o_ * angular_velocity;
        // angular_velocity[0] = 0;
        // angular_velocity[1] = 0;

        pintegration_->propagate(dt, linear_velocity, angular_velocity);

        last_time_ = imu_msg->header.stamp.toSec();
#if USE_IMU_POSE
        last_eular_ = eular_angles;
        last_q_ = q;
#endif
    }

    //publish result
    if (pintegration_)
    {
        //cout<<"p:"<<pOdomIntegration->delta_p.transpose()<<endl;
        std_msgs::Header header = points_msg->header;
        // header.frame_id = "world";

        nav_msgs::Odometry odometry;
        odometry.header = header;
        //odometry.header.frame_id = "world";
        //odometry.child_frame_id = "world";
        odometry.twist = odom_msg->twist;
        odometry.pose.pose.position.x = (float)pintegration_->delta_p.x();
        odometry.pose.pose.position.y = (float)pintegration_->delta_p.y();
        odometry.pose.pose.position.z = (float)pintegration_->delta_p.z();
        odometry.pose.pose.orientation.x = (float)Quaterniond(pintegration_->delta_q).x();
        odometry.pose.pose.orientation.y = (float)Quaterniond(pintegration_->delta_q).y();
        odometry.pose.pose.orientation.z = (float)Quaterniond(pintegration_->delta_q).z();
        odometry.pose.pose.orientation.w = (float)Quaterniond(pintegration_->delta_q).w();
        pub_odom_.publish(odometry);

        //path
        path_.header = header;
        path_.header.frame_id = "camera_init";
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.pose = odometry.pose.pose;
        path_.poses.push_back(pose_stamped);
        pub_path_.publish(path_);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "imu_odom");
    ros::NodeHandle n("~");

    ROS_INFO("\033[1;32m---->\033[0m imu odometry Started.");

    i2o_ = Eigen::Matrix3d::Identity();
    // i2o_ << 0.99990731, -0.010571127, -0.008578172,
    //     0.010883288, 0.99924862, 0.03719838,
    //     0.0081784986, -0.037288293, 0.99927109;

    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(n, "/odom_chassis", 1);
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub(n, "/imu0", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> points_sub(n, "/rslidar_points", 1);
    typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::Imu, sensor_msgs::PointCloud2> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(100), odom_sub, imu_sub, points_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2, _3));

    pub_odom_ = n.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 1);
    pub_path_ = n.advertise<nav_msgs::Path>("/fusion_path", 1);

    ros::spin();

    return 0;
}
