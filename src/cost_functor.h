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
  bool operator()(const T* const translation_ptr, const T* const quaternion_ptr,
                  const T* const point_ptr, const T* const k1_ptr,
                  const T* const k2_ptr, const T* const fx_ptr,
                  const T* const fy_ptr, const T* const cx_ptr,
                  const T* const cy_ptr, T* residual_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(translation_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> q(quaternion_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> X(point_ptr);

    Eigen::Matrix<T, 3, 1> warped_point = q * X + t;

    if (warped_point.z() < T(0.01)) return false;

    Eigen::Matrix<T, 3, 1> image_coordinate;
    image_coordinate.x() = warped_point.x() / warped_point.z();
    image_coordinate.y() = warped_point.y() / warped_point.z();
    image_coordinate.z() = T(1.0);

    T r2 = image_coordinate.x() * image_coordinate.x() +
           image_coordinate.y() * image_coordinate.y();

    T D = T(1.0) + *k1_ptr * r2 + *k2_ptr * r2 * r2;

    if (!residual_ptr) {
      return false;
    }
    residual_ptr[0] =
        *fx_ptr * image_coordinate.x() / D + *cx_ptr - T(distorted_pixel_.x());
    residual_ptr[1] =
        *fy_ptr * image_coordinate.y() / D + *cy_ptr - T(distorted_pixel_.y());

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector2d& distorted_pixel) {
    constexpr int kDimResidual = 2;
    constexpr int kDimTranslation = 3;
    constexpr int kDimQuaternion = 4;
    constexpr int kDimPoint = 3;
    constexpr int kDimRadialDistortionK1 = 1;
    constexpr int kDimRadialDistortionK2 = 1;
    constexpr int kDimFocalLengthX = 1;
    constexpr int kDimFocalLengthY = 1;
    constexpr int kDimImageCenterX = 1;
    constexpr int kDimImageCenterY = 1;
    return new ceres::AutoDiffCostFunction<
        DistortedReprojectionErrorCostFunctor, kDimResidual, kDimTranslation,
        kDimQuaternion, kDimPoint, kDimRadialDistortionK1,
        kDimRadialDistortionK2, kDimFocalLengthX, kDimFocalLengthY,
        kDimImageCenterX, kDimImageCenterY>(
        new DistortedReprojectionErrorCostFunctor(distorted_pixel));
  }

 private:
  const Eigen::Vector2d distorted_pixel_;
};

#endif  // COST_FUNCTOR_H_