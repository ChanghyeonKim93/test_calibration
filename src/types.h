#ifndef TYPES_H_
#define TYPES_H_

#include <map>
#include <memory>
#include <vector>

#include "Eigen/Dense"

using Pose = Eigen::Isometry3d;
using Mat3x3 = Eigen::Matrix3d;
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Quaternion = Eigen::Quaterniond;

struct Observation {
  int board_point_id{-1};
  Vec2 distorted_pixel{Vec2::Zero()};
};

struct CameraInfo {
  int id{-1};
  int image_height{-1};
  int image_width{-1};
  double intrinsic_parameter[4] = {-1.0};   // fx fy cx cy
  double distortion_parameter[4] = {-1.0};  // k1 k2 p1 p2
  Pose extrinsic_pose{Pose::Identity()};
};
using CameraInfoPtr = std::shared_ptr<CameraInfo>;

struct CameraFrame {
  int id{-1};
  CameraInfoPtr camera_info_ptr{nullptr};
  std::vector<Observation> observation_list;
};

struct LidarInfo {
  Pose extrinsic_pose{Pose::Identity()};
};
using LidarInfoPtr = std::shared_ptr<LidarInfo>;
struct LidarFrame {
  int id{-1};
  LidarInfoPtr lidar_info_ptr{nullptr};
  std::vector<Vec3> plane_point_list;
};

struct Baselink {
  int id{-1};
  double timestamp{0.0};
  Pose pose{Pose::Identity()};
  std::map<int, CameraFrame> camera_frame_map;
};

#endif  // TYPES_H_