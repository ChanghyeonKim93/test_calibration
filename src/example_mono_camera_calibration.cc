#include <iostream>
#include <random>
#include <stdexcept>

#include "Eigen/Dense"

#include "cost_functor.h"
#include "types.h"
#include "utilities.h"

namespace {

std::random_device rd;
std::uniform_real_distribution uniform_dist(0.0, 1.0);

};  // namespace

std::map<int, Vec3> GenerateBoardPointMap() {
  std::map<int, Vec3> board_point_map{
      {0, {0.0, 0.0, 0.0}}, {1, {0.0, 0.1, 0.0}}, {2, {0.0, 0.2, 0.0}},
      {3, {0.1, 0.0, 0.0}}, {4, {0.1, 0.1, 0.0}}, {5, {0.1, 0.2, 0.0}},
      {6, {0.2, 0.0, 0.0}}, {7, {0.2, 0.1, 0.0}}, {8, {0.2, 0.2, 0.0}}};
  return board_point_map;
}

std::map<int, CameraPtr> GenerateCameraMap() {
  std::map<int, CameraPtr> camera_map;
  CameraPtr camera_0_ptr = std::make_shared<Camera>();
  camera_0_ptr->id = 0;
  camera_0_ptr->image_height = 480;
  camera_0_ptr->image_width = 640;
  camera_0_ptr->intrinsic_parameter[0] = 451.5;   // fx
  camera_0_ptr->intrinsic_parameter[1] = 452.3;   // fy
  camera_0_ptr->intrinsic_parameter[2] = 320.4;   // cx
  camera_0_ptr->intrinsic_parameter[3] = 240.7;   // cy
  camera_0_ptr->distortion_parameter[0] = -0.21;  // k1
  camera_0_ptr->distortion_parameter[1] = 0.04;   // k2
  camera_0_ptr->distortion_parameter[2] = 0.0;    // p1
  camera_0_ptr->distortion_parameter[3] = 0.0;    // p2
  camera_0_ptr->extrinsic_pose.setIdentity();
  camera_map.insert({camera_0_ptr->id, camera_0_ptr});

  return camera_map;
};

std::vector<Pose> GenerateBaselinkPoseList() {
  constexpr size_t kNumPose{10};

  Pose baselink_pose{Pose::Identity()};
  baselink_pose.translation().x() = -1.0;
  baselink_pose.translation().z() = -2.0;

  std::vector<Pose> baselink_pose_list;
  baselink_pose_list.push_back(baselink_pose);
  for (size_t i = 1; i < kNumPose; ++i) {
    baselink_pose.translation().x() += 0.2;
    baselink_pose.translation().y() += 0.05;
    baselink_pose_list.push_back(baselink_pose);
  }
  return baselink_pose_list;
}

int main() {
  // Generate simulation data
  const auto true_point_map = GenerateBoardPointMap();
  const auto true_camera_map = GenerateCameraMap();
  const auto true_baselink_pose_list = GenerateBaselinkPoseList();
  std::map<int, Baselink> true_baselink_map;
  for (size_t baselink_id = 0; baselink_id < true_baselink_pose_list.size();
       ++baselink_id) {
    const auto& pose = true_baselink_pose_list.at(baselink_id);
    Baselink baselink;
    baselink.id = baselink_id;
    baselink.timestamp = baselink_id * 0.1;
    baselink.pose = pose;
    baselink.camera_frame_map.insert({0, {0, true_camera_map.at(0), {}}});
    true_baselink_map.insert({baselink_id, baselink});
  }
  for (auto& [baselink_id, baselink] : true_baselink_map) {
    const Pose& baselink_pose = baselink.pose;
    for (auto& [frame_id, camera_frame] : baselink.camera_frame_map) {
      const auto& camera_ptr = camera_frame.camera_ptr;
      for (const auto& [point_id, true_point] : true_point_map) {
        const auto warped_point = camera_ptr->extrinsic_pose.inverse() *
                                  baselink_pose.inverse() * true_point;
        if (warped_point.z() < 0.01) continue;
        const auto distorted_projected_pixel =
            ProjectToDistortedPixel(warped_point, camera_ptr);
        if (!IsInImage(distorted_projected_pixel, camera_ptr->image_height,
                       camera_ptr->image_width))
          continue;

        Observation observation{point_id, distorted_projected_pixel};
        camera_frame.observation_list.push_back(observation);
      }
      std::cerr << "baselink_id, frame_id : " << baselink_id << ", " << frame_id
                << "-> num observation: "
                << camera_frame.observation_list.size() << std::endl;
    }
  }

  // Initialization realistic data
  // Note: intentionally distort values
  std::map<int, Vec3> point_map;
  for (const auto& [point_id, true_point] : true_point_map) {
    Vec3 point = true_point;
    point.x() += uniform_dist(rd) * 0.05;
    point.y() += uniform_dist(rd) * 0.05;
    point.z() += uniform_dist(rd) * 0.1;
    point_map.insert({point_id, point});
  }

  std::map<int, CameraPtr> camera_map;
  for (const auto& [camera_id, true_camera_ptr] : true_camera_map) {
    CameraPtr camera_ptr = std::make_shared<Camera>(*true_camera_ptr);
    camera_ptr->extrinsic_pose.setIdentity();
    // camera_ptr->intrinsic_parameter[0] += uniform_dist(rd) * 10.0;
    camera_ptr->intrinsic_parameter[1] += uniform_dist(rd) * 10.0;
    camera_ptr->intrinsic_parameter[2] += uniform_dist(rd) * 5.0;
    camera_ptr->intrinsic_parameter[3] += uniform_dist(rd) * 5.0;
    camera_ptr->distortion_parameter[0] = 0.0;
    camera_ptr->distortion_parameter[1] = 0.0;
    camera_ptr->distortion_parameter[2] = 0.0;
    camera_ptr->distortion_parameter[3] = 0.0;
    camera_map.insert({camera_id, camera_ptr});
  }

  std::map<int, Baselink> baselink_map;
  for (const auto& [baselink_id, true_baselink] : true_baselink_map) {
    Baselink baselink;
    baselink = true_baselink;
    if (baselink_id != 0) {
      Pose random_deviation{Pose::Identity()};
      random_deviation.translation().x() += uniform_dist(rd) * 0.05;
      random_deviation.translation().y() += uniform_dist(rd) * 0.05;
      random_deviation.translation().z() += uniform_dist(rd) * 0.05;
      random_deviation.linear() =
          Quaternion(1.0, uniform_dist(rd) * 0.05, uniform_dist(rd) * 0.05,
                     uniform_dist(rd) * 0.05)
              .normalized()
              .toRotationMatrix();
      baselink.pose = baselink.pose * random_deviation;
    }
    baselink_map.insert({baselink_id, baselink});
  }

  // Generate parameter map for optimization
  using PointParameter = Vec3;
  struct PoseParameter {
    Vec3 t;
    Quaternion q;
  };
  struct CameraParameter {
    double intrinsic[4];
    double distortion[4];
    Vec3 t;
    Quaternion q;
  };
  std::map<int, PointParameter> param_map_for_point;
  for (const auto& [point_id, point] : point_map) {
    param_map_for_point.insert({point_id, point});
  }

  std::map<int, PoseParameter> param_map_for_baselink_pose;
  for (const auto& [baselink_id, baselink] : baselink_map) {
    const Pose inverse_pose = baselink.pose.inverse();
    param_map_for_baselink_pose.insert(
        {baselink_id,
         {inverse_pose.translation(),
          Quaternion(inverse_pose.rotation()).normalized()}});
  }

  std::map<int, CameraParameter> param_map_for_camera_parameter;
  for (const auto& [camera_id, camera_ptr] : camera_map) {
    CameraParameter camera_param;
    for (int i = 0; i < 4; ++i)
      camera_param.intrinsic[i] = camera_ptr->intrinsic_parameter[i];

    for (int i = 0; i < 4; ++i)
      camera_param.distortion[i] = camera_ptr->distortion_parameter[i];

    const Pose inverse_pose = camera_ptr->extrinsic_pose.inverse();
    camera_param.t = inverse_pose.translation();
    camera_param.q = Quaternion(inverse_pose.rotation()).normalized();
    param_map_for_camera_parameter.insert({camera_id, camera_param});
  }

  // Optimize
  ceres::Solver::Options options;
  options.max_num_iterations = 400;
  options.function_tolerance = 1e-7;
  options.gradient_tolerance = 1e-7;
  options.parameter_tolerance = 1e-7;
  ceres::Solver::Summary summary;
  ceres::Problem ceres_problem;

  // Add cost functor for each observation
  for (auto& [baselink_id, baselink] : baselink_map) {
    auto& baselink_t = param_map_for_baselink_pose.at(baselink_id).t;
    auto& baselink_q = param_map_for_baselink_pose.at(baselink_id).q;
    for (auto& [frame_id, camera_frame] : baselink.camera_frame_map) {
      const CameraPtr& camera_ptr = camera_frame.camera_ptr;
      auto& camera_extrinsic_t =
          param_map_for_camera_parameter.at(camera_ptr->id).t;
      auto& camera_extrinsic_q =
          param_map_for_camera_parameter.at(camera_ptr->id).q;
      auto& intrinsics =
          param_map_for_camera_parameter.at(camera_ptr->id).intrinsic;
      auto& distortions =
          param_map_for_camera_parameter.at(camera_ptr->id).distortion;
      for (const auto& [point_id, distorted_pixel] :
           camera_frame.observation_list) {
        auto& point = param_map_for_point.at(point_id);
        ceres::CostFunction* cost_function =
            DistortedReprojectionErrorCostFunctor::Create(distorted_pixel);
        ceres::LossFunction* loss_function = nullptr;
        ceres_problem.AddResidualBlock(
            cost_function, loss_function, baselink_t.data(),
            baselink_q.coeffs().data(), camera_extrinsic_t.data(),
            camera_extrinsic_q.coeffs().data(), point.data(), intrinsics,
            distortions);
      }
    }
  }

  // Fix some parameters
  ceres_problem.SetParameterBlockConstant(
      param_map_for_baselink_pose.at(0).t.data());
  ceres_problem.SetParameterBlockConstant(
      param_map_for_baselink_pose.at(0).q.coeffs().data());
  ceres_problem.SetParameterBlockConstant(
      param_map_for_camera_parameter.at(0).t.data());
  ceres_problem.SetParameterBlockConstant(
      param_map_for_camera_parameter.at(0).q.coeffs().data());
  for (auto& [camera_id, camera_parameter] : param_map_for_camera_parameter) {
    ceres::SubsetParameterization* subset_parameterization =
        new ceres::SubsetParameterization(4, {0});
    ceres_problem.SetParameterization(camera_parameter.intrinsic,
                                      subset_parameterization);
  }

  // Use parametrization
  for (auto& [baselink_id, baselink_param] : param_map_for_baselink_pose) {
    ceres_problem.SetParameterization(
        baselink_param.q.coeffs().data(),
        new ceres::EigenQuaternionParameterization());
  }
  for (auto& [camera_id, camera_parameter] : param_map_for_camera_parameter) {
    ceres_problem.SetParameterization(
        camera_parameter.q.coeffs().data(),
        new ceres::EigenQuaternionParameterization());
  }

  ceres::Solve(options, &ceres_problem, &summary);

  // Show the results
  std::cerr << summary.FullReport() << std::endl;

  std::cerr << "[True] fx,fy,cx,cy : "
            << true_camera_map.at(0)->intrinsic_parameter[0] << ","
            << true_camera_map.at(0)->intrinsic_parameter[1] << ","
            << true_camera_map.at(0)->intrinsic_parameter[2] << ","
            << true_camera_map.at(0)->intrinsic_parameter[3] << std::endl;
  std::cerr << "[Est ] fx,fy,cx,cy : "
            << param_map_for_camera_parameter.at(0).intrinsic[0] << ","
            << param_map_for_camera_parameter.at(0).intrinsic[1] << ","
            << param_map_for_camera_parameter.at(0).intrinsic[2] << ","
            << param_map_for_camera_parameter.at(0).intrinsic[3] << std::endl;

  std::cerr << "[True] k1,k2,p1,p2: "
            << true_camera_map.at(0)->distortion_parameter[0] << ", "
            << true_camera_map.at(0)->distortion_parameter[1] << ", "
            << true_camera_map.at(0)->distortion_parameter[2] << ", "
            << true_camera_map.at(0)->distortion_parameter[3] << std::endl;
  std::cerr << "[Est ] k1,k2,p1,p2: "
            << param_map_for_camera_parameter.at(0).distortion[0] << ", "
            << param_map_for_camera_parameter.at(0).distortion[1] << ", "
            << param_map_for_camera_parameter.at(0).distortion[2] << ", "
            << param_map_for_camera_parameter.at(0).distortion[3] << std::endl;

  return 0;
}