#ifndef COST_FUNCTOR_H_
#define COST_FUNCTOR_H_

#include "Eigen/Dense"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

class DistortedReprojectionErrorCostFunctor {
 public:
  DistortedReprojectionErrorCostFunctor(const Eigen::Vector2d& distorted_pixel)
      : distorted_pixel_(distorted_pixel) {}

  template <typename T>
  bool operator()(const T* const baselink_translation_ptr,
                  const T* const baselink_quaternion_ptr,
                  const T* const relative_translation_from_baselink_ptr,
                  const T* const relative_quaternion_from_baselink_ptr,
                  const T* const point_ptr,
                  const T* const intrinsic_parameter_ptr,
                  const T* const distortion_parameter_ptr,
                  T* residual_ptr) const {
    if (!residual_ptr) return false;

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> baselink_t(
        baselink_translation_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> baselink_q(baselink_quaternion_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> camera_t(
        relative_translation_from_baselink_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> camera_q(
        relative_quaternion_from_baselink_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> X(point_ptr);

    Eigen::Matrix<T, 3, 1> warped_point =
        camera_q * (baselink_q * X + baselink_t) + camera_t;

    if (warped_point.z() < T(0.01)) {
      std::cerr << "point is not front of the camera\n";
      return false;
    }

    T inverse_z = T(1.0) / warped_point.z();
    T image_coordinate_x = warped_point.x() * inverse_z;
    T image_coordinate_y = warped_point.y() * inverse_z;

    T squared_x = image_coordinate_x * image_coordinate_x;
    T squared_y = image_coordinate_y * image_coordinate_y;
    T xy = image_coordinate_x * image_coordinate_y;
    T squared_r = squared_x + squared_y;

    T fx = intrinsic_parameter_ptr[0];
    T fy = intrinsic_parameter_ptr[1];
    T cx = intrinsic_parameter_ptr[2];
    T cy = intrinsic_parameter_ptr[3];
    T k1 = distortion_parameter_ptr[0];
    T k2 = distortion_parameter_ptr[1];
    T p1 = distortion_parameter_ptr[2];
    T p2 = distortion_parameter_ptr[3];

    T radial_distortion_factor =
        T(1.0) + k1 * squared_r + k2 * squared_r * squared_r;
    T tangential_distortion_factor_x =
        T(2.0) * p1 * xy + p2 * (squared_r + T(2.0) * squared_x);
    T tangential_distortion_factor_y =
        T(2.0) * p2 * xy + p1 * (squared_r + T(2.0) * squared_y);

    T distorted_x = image_coordinate_x * radial_distortion_factor +
                    tangential_distortion_factor_x;
    T distorted_y = image_coordinate_y * radial_distortion_factor +
                    tangential_distortion_factor_y;
    residual_ptr[0] = fx * distorted_x + cx - T(distorted_pixel_.x());
    residual_ptr[1] = fy * distorted_y + cy - T(distorted_pixel_.y());

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector2d& distorted_pixel) {
    constexpr int kDimResidual = 2;
    constexpr int kDimTranslation = 3;
    constexpr int kDimQuaternion = 4;
    constexpr int kDimPoint = 3;
    constexpr int kDimIntrinsicParameter = 4;
    constexpr int kDimDistortionParameter = 4;
    return new ceres::AutoDiffCostFunction<
        DistortedReprojectionErrorCostFunctor, kDimResidual, kDimTranslation,
        kDimQuaternion, kDimTranslation, kDimQuaternion, kDimPoint,
        kDimIntrinsicParameter, kDimDistortionParameter>(
        new DistortedReprojectionErrorCostFunctor(distorted_pixel));
  }

 private:
  const Eigen::Vector2d distorted_pixel_;
};

#endif  // COST_FUNCTOR_H_