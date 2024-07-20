#ifndef POSE_GRAPH_OPTIMIZER_COST_FUNCTOR_H_
#define POSE_GRAPH_OPTIMIZER_COST_FUNCTOR_H_

#include "optimizer/types.h"

#include "Eigen/Dense"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace optimizer {

template <typename T>
void ConvertToHamiltonQuaternion(const T* const q, T* hamilton_q) {
  hamilton_q[1] = q[0];
  hamilton_q[2] = q[1];
  hamilton_q[3] = q[2];
  hamilton_q[0] = q[3];
}
template <typename T>
void ConvertToHamiltonQuaternion(T* q) {
  T qx = q[0];
  q[0] = q[3];
  q[1] = qx;
  q[2] = q[1];
  q[3] = q[2];
}

template <typename T>
void ConvertToJPLQuaternion(const T* const q, T* jpl_q) {
  jpl_q[0] = q[1];
  jpl_q[1] = q[2];
  jpl_q[2] = q[3];
  jpl_q[3] = q[0];
}

template <typename T>
void ConvertToJPLQuaternion(T* q) {
  T qw = q[0];
  q[0] = q[1];
  q[1] = q[2];
  q[2] = q[3];
  q[3] = qw;
}

/// @brief Relative pose cost functor.
///  [ p01 ]  = [ R(q0)^T * (p1 - p0) ]
///  [ q01 ]  = [ q0^{-1] * q1        ]
///
///   error = [ p01 - \hat{p01}                 ]
///           [ 2.0 * Vec(q01 * \hat{q01}^{-1}) ]
class RelativePoseCostFunctor {
 public:
  RelativePoseCostFunctor(const Eigen::Isometry3d& relative_pose_01,
                          const Eigen::Matrix<double, 6, 6>& sqrt_information)
      : relative_p01_(relative_pose_01.translation()),
        relative_q01_(Eigen::Quaterniond(relative_pose_01.rotation())),
        sqrt_information_(sqrt_information) {}

  template <typename T>
  bool operator()(const T* const p0_ptr, const T* const rvec0_ptr,
                  const T* const p1_ptr, const T* const rvec1_ptr,
                  T* residual_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p0(p0_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p1(p1_ptr);

    T q0_tmp[4];
    T q1_tmp[4];
    ceres::AngleAxisToQuaternion(rvec0_ptr, q0_tmp);
    ceres::AngleAxisToQuaternion(rvec1_ptr, q1_tmp);
    ConvertToJPLQuaternion<T>(q0_tmp);
    ConvertToJPLQuaternion<T>(q1_tmp);
    Eigen::Map<const Eigen::Quaternion<T>> q0(q0_tmp);
    Eigen::Map<const Eigen::Quaternion<T>> q1(q1_tmp);

    Eigen::Quaternion<T> error_q =
        relative_q01_.template cast<T>().conjugate() * (q0.conjugate() * q1);
    Eigen::Matrix<T, 3, 1> error_p =
        q0 * relative_p01_.template cast<T>() - (p1 - p0);

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residual_ptr);
    residual.template block<3, 1>(0, 0) = error_p;
    residual.template block<3, 1>(3, 0) = T(2.0) * error_q.vec();

    // // Scale the residuals by the measurement uncertainty.
    // // residuals.applyOnTheLeft(sqrt_information_.template cast<T>());
    // residual = sqrt_information_.cast<T> * residual;
    (void)sqrt_information_;

    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Isometry3d& relative_pose_01,
      const Eigen::Matrix<double, 6, 6>& sqrt_information) {
    return new ceres::AutoDiffCostFunction<RelativePoseCostFunctor, 6, 3, 3, 3,
                                           3>(
        new RelativePoseCostFunctor(relative_pose_01, sqrt_information));
  }

 private:
  const Eigen::Vector3d relative_p01_;
  const Eigen::Quaterniond relative_q01_;
  const Eigen::Matrix<double, 6, 6> sqrt_information_;
  // The square root of the measurement information matrix.
};

}  // namespace optimizer

#endif  // POSE_GRAPH_OPTIMIZER_COST_FUNCTOR_H_