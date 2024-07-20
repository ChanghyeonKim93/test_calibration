#include <iostream>
#include <map>
#include <vector>

#include "Eigen/Dense"

#include "cost_functor.h"

using Pose = Eigen::Isometry3d;
using Mat3x3 = Eigen::Matrix3d;
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;

struct Observation {
  size_t id{0};
  Vec2 distorted_pixel;
};

struct Frame {
  int id{-1};
  Pose inverse_pose{Pose::Identity()};
  std::vector<Observation> observation_list;
};

inline bool IsInImage(const Vec2& pixel, const int height, const int width) {
  return (pixel.x() >= 0.0 && pixel.x() < width && pixel.y() >= 0.0 &&
          pixel.y() < height);
}

Vec3 ConvertToImageCoordinate(const Vec3& point) {
  Vec3 image_coordinate;
  image_coordinate.x() = point.x() / point.z();
  image_coordinate.y() = point.y() / point.z();
  image_coordinate.z() = 1.0;
  return image_coordinate;
}

Vec2 ProjectToPixelCoordinate(const Vec3& point, const double fx,
                              const double fy, const double cx,
                              const double cy) {
  const double inverse_z = 1.0 / point.z();
  Vec2 pixel;
  pixel.x() = fx * point.x() * inverse_z + cx;
  pixel.y() = fy * point.y() * inverse_z + cy;
  return pixel;
}

Vec2 ProjectToDistortedPixel(const Vec3& point, const double fx,
                             const double fy, const double cx, const double cy,
                             const double k1, const double k2) {
  Vec3 image_coordinate;
  image_coordinate.x() = point.x() / point.z();
  image_coordinate.y() = point.y() / point.z();
  image_coordinate.z() = 1.0;
  const double xu = image_coordinate.x();
  const double yu = image_coordinate.y();
  const double r2 = xu * xu + yu * yu;
  const double D = 1.0 + k1 * r2 + k2 * r2 * r2;
  const double inv_D = 1.0 / D;
  const double xd = xu * inv_D;
  const double yd = yu * inv_D;
  Vec2 distorted_pixel{fx * xd + cx, fy * yd + cy};
  return distorted_pixel;
}

int main() {
  constexpr size_t kNumPose = 10;
  constexpr int kImageWidth = 640;
  constexpr int kImageHeight = 480;
  constexpr double kFx = 451.5;
  constexpr double kFy = 452.3;
  constexpr double kCx = kImageWidth * 0.5 + 0.4;
  constexpr double kCy = kImageHeight * 0.5 - 0.7;
  constexpr double kK1 = -0.28340811;
  constexpr double kK2 = 0.07395907;

  std::vector<Vec3> point_list;
  point_list.push_back({0.0, 0.0, 0.0});
  point_list.push_back({0.0, 0.1, 0.0});
  point_list.push_back({0.0, 0.2, 0.0});
  point_list.push_back({0.1, 0.0, 0.0});
  point_list.push_back({0.1, 0.1, 0.0});
  point_list.push_back({0.1, 0.2, 0.0});
  point_list.push_back({0.2, 0.0, 0.0});
  point_list.push_back({0.2, 0.1, 0.0});
  point_list.push_back({0.2, 0.2, 0.0});

  Pose pose{Pose::Identity()};
  pose.translation().x() = -1.0;
  pose.translation().z() = -2.0;

  std::vector<Pose> pose_list;
  pose_list.push_back(pose);
  for (size_t i = 1; i < kNumPose; ++i) {
    pose.translation().x() += 0.2;
    pose.translation().y() += 0.1;
    pose_list.push_back(pose);
  }

  // Generate frame lise
  std::vector<Frame> frame_list;
  for (size_t index = 0; index < kNumPose; ++index) {
    const auto& pose = pose_list.at(index);
    Frame frame;
    frame.id = index;
    frame.inverse_pose = pose.inverse();
    frame_list.push_back(frame);
  }

  // Project point
  for (auto& frame : frame_list) {
    for (size_t point_index = 0; point_index < point_list.size();
         ++point_index) {
      const auto& point = point_list.at(point_index);
      const auto warped_point = frame.inverse_pose * point;
      if (warped_point.z() < 0.1) continue;
      const auto distorted_projected_pixel =
          ProjectToDistortedPixel(warped_point, kFx, kFy, kCx, kCy, kK1, kK2);
      if (!IsInImage(distorted_projected_pixel, kImageHeight, kImageWidth))
        continue;
      Observation observation{point_index, distorted_projected_pixel};
      frame.observation_list.push_back(observation);
    }

    std::cerr << "Id: " << frame.id
              << ", num observation: " << frame.observation_list.size()
              << std::endl;
  }

  // Optimize
  ceres::Solver::Options options;
  options.max_num_iterations = 400;
  options.function_tolerance = 1e-7;
  options.gradient_tolerance = 1e-7;
  options.parameter_tolerance = 1e-7;
  ceres::Solver::Summary summary;
  ceres::Problem ceres_problem;

  struct PoseParameter {
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
  };
  std::map<int, PoseParameter> pose_param_map;
  double k1 = 0.0;
  double k2 = 0.0;
  double fx = 451.5;
  double fy = 200.0;
  double cx = 300;
  double cy = 200;
  for (auto& frame : frame_list) {
    pose_param_map.insert(
        {static_cast<size_t>(frame.id),
         {frame.inverse_pose.translation(),
          Eigen::Quaterniond(frame.inverse_pose.rotation())}});
  }
  for (auto& frame : frame_list) {
    auto& t = pose_param_map.at(frame.id).t;
    auto& q = pose_param_map.at(frame.id).q;
    for (const auto& [point_index, distorted_pixel] : frame.observation_list) {
      auto& point = point_list.at(point_index);
      ceres::CostFunction* cost_function =
          DistortedReprojectionErrorCostFunctor::Create(distorted_pixel);
      ceres::LossFunction* loss_function = nullptr;
      ceres_problem.AddResidualBlock(cost_function, loss_function, t.data(),
                                     q.coeffs().data(), point.data(), &k1, &k2,
                                     &fx, &fy, &cx, &cy);
    }
    if (frame.id == 0) {
      ceres_problem.SetParameterBlockConstant(t.data());
      ceres_problem.SetParameterBlockConstant(q.coeffs().data());
    }
    ceres_problem.SetParameterization(
        q.coeffs().data(), new ceres::EigenQuaternionParameterization());
  }
  ceres_problem.SetParameterBlockConstant(&fx);
  // ceres_problem.SetParameterBlockConstant(&fy);
  // ceres_problem.SetParameterBlockConstant(&cx);
  // ceres_problem.SetParameterBlockConstant(&cy);

  ceres::Solve(options, &ceres_problem, &summary);
  std::cerr << summary.FullReport() << std::endl;

  std::cerr << "fx,fy,cx,cy : " << kFx << ", " << kFy << ", " << kCx << ", "
            << kCy << std::endl;
  std::cerr << "fx,fy,cx,cy : " << fx << ", " << fy << ", " << cx << ", " << cy
            << std::endl;

  std::cerr << "k1,k2: " << kK1 << ", " << kK2 << std::endl;
  std::cerr << "k1,k2: " << k1 << ", " << k2 << std::endl;

  return 0;
}