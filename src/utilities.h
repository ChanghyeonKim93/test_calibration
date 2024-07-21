#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <stdexcept>

#include "types.h"

inline std::pair<Vec3, Quaternion> ConvertToTranslationAndQuaternion(
    const Pose& pose) {
  std::pair<Vec3, Quaternion> translation_and_quaternion;
  translation_and_quaternion.first = pose.translation();
  translation_and_quaternion.second = Quaternion(pose.rotation()).normalized();
  return translation_and_quaternion;
}

inline bool IsInImage(const Vec2& pixel, const int image_height,
                      const int image_width) {
  return (pixel.x() >= 0.0 && pixel.x() < image_width && pixel.y() >= 0.0 &&
          pixel.y() < image_height);
}

Vec2 ProjectToDistortedPixel(const Vec3& point, const double fx,
                             const double fy, const double cx, const double cy,
                             const double k1, const double k2, const double p1,
                             const double p2) {
  constexpr double kMinDepth{0.01};  // [m]
  if (point.z() < kMinDepth)
    throw std::invalid_argument("point.z() < kMinDepth");

  const double inverse_z = 1.0 / point.z();
  Vec3 image_coordinate{point.x() * inverse_z, point.y() * inverse_z, 1.0};
  const double xu = image_coordinate.x();  // undistorted image coordinate x
  const double yu = image_coordinate.y();  // undistorted image coordinate y
  const double squared_xu = xu * xu;
  const double squared_yu = yu * yu;
  const double xuyu = xu * yu;
  const double squared_r = squared_xu + squared_yu;
  const double radial_distortion_factor =
      1.0 + k1 * squared_r + k2 * squared_r * squared_r;
  const double tangential_distortion_factor_x =
      2.0 * p1 * xuyu + p2 * (squared_r + 2.0 * squared_xu);
  const double tangential_distortion_factor_y =
      2.0 * p2 * xuyu + p1 * (squared_r + 2.0 * squared_yu);
  const double xd =
      xu * radial_distortion_factor +
      tangential_distortion_factor_x;  // distorted image coordinate x
  const double yd =
      yu * radial_distortion_factor +
      tangential_distortion_factor_y;  // distorted image coordinate y
  const Vec2 distorted_pixel{fx * xd + cx, fy * yd + cy};

  return distorted_pixel;
}

Vec2 ProjectToDistortedPixel(const Vec3& point,
                             const CameraInfoPtr& camera_ptr) {
  if (camera_ptr == nullptr)
    throw std::invalid_argument("camera_ptr == nullptr");

  const double fx = camera_ptr->intrinsic_parameter[0];
  const double fy = camera_ptr->intrinsic_parameter[1];
  const double cx = camera_ptr->intrinsic_parameter[2];
  const double cy = camera_ptr->intrinsic_parameter[3];
  const double k1 = camera_ptr->distortion_parameter[0];
  const double k2 = camera_ptr->distortion_parameter[1];
  const double p1 = camera_ptr->distortion_parameter[2];
  const double p2 = camera_ptr->distortion_parameter[3];

  return ProjectToDistortedPixel(point, fx, fy, cx, cy, k1, k2, p1, p2);
}

#endif  // UTILITIES_H_