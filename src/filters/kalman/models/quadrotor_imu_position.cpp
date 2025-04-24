#include "roblib/filters/models/quadrotor_imu_position.hpp"
#include "roblib/filters/kalman/unscented_kalman_filter.hpp"

/** 
 * @brief The Quadrotor class implements the quadrotor model for the Unscented Kalman filter 
 * This is the default constructor, it does not do anything
 */
Quadrotor::Quadrotor()
{
}

Eigen::Matrix<Quadrotor::TYPE, Quadrotor::N_STATES, 1>
Quadrotor::stateTransitionFunction(Eigen::Matrix<Quadrotor::TYPE, Quadrotor::N_AUGMENTED_PROCESS_STATE, 1> x,
                             const Eigen::Matrix<Quadrotor::TYPE, Quadrotor::N_CONTROLS, 1> u,
                                   Quadrotor::TYPE dt)
{
  // !< Update dt
  _dt = dt;

  /**
   * Precomputed to reduce computation time
   */

  _Rgyro_quat.w() = 0.0;
  Eigen::Matrix<TYPE, 3, 1> u_w;
  u_w << (u(3) - x(13) - x(19)), (u(4) - x(14) - x(20)),
      (u(5) - x(15) - x(21));

  TYPE norm_u_w = u_w.norm();

  TYPE sin_norm = sin(norm_u_w * dt / 2.0);
  TYPE cos_norm = cos(norm_u_w * dt / 2.0);
  _Rgyro_quat.w() = norm_u_w < 1e-5 ? 1.0 : cos_norm;
  _Rgyro_quat.x() = norm_u_w < 1e-5 ? 0.0 : u_w(0) * sin_norm / norm_u_w;
  _Rgyro_quat.y() = norm_u_w < 1e-5 ? 0.0 : u_w(1) * sin_norm / norm_u_w;
  _Rgyro_quat.z() = norm_u_w < 1e-5 ? 0.0 : u_w(2) * sin_norm / norm_u_w;

  _quat_current.w() = x(6);
  _quat_current.x() = x(7);
  _quat_current.y() = x(8);
  _quat_current.z() = x(9);

  _x_dot.setZero(); //!< Initialize the state transition function to zero

  /**
   * Position dot = Velocity
   */
  _x_dot(0) = x(3);
  _x_dot(1) = x(4);
  _x_dot(2) = x(5);

  /**
   *  Velocity dot = Rotation_matrix * Body Acceleration + Gravity
   * Transform the acceleration from the body frame to the world frame
   *                          currentQuat  *   measured acc body   - bias acc
   * body       - noise acc body
   */
  _x_dot.block<3, 1>(3, 0) =
      _quat_current *
      (u.block<3, 1>(0, 0) - x.block<3, 1>(10, 0) - x.block<3, 1>(16, 0));
  _x_dot(5) += _gravity;

  /**
   * Orientation dot = Rotation_matrix * Body Rotation_rate
   */
  Eigen::Quaternion<TYPE> _quat_new;
  _quat_new = _quat_current * _Rgyro_quat;
  _quat_new.normalize();

  _x_dot(6) = _quat_new.w();
  _x_dot(7) = _quat_new.x();
  _x_dot(8) = _quat_new.y();
  _x_dot(9) = _quat_new.z();

  /**
   * Accel Bias dot = noise bias accelerometer
   */
  _x_dot(10) = x(22);
  _x_dot(11) = x(23);
  _x_dot(12) = x(24);

  /**
   * Gyro Bias dot = noise bias gyroscope
   */

  _x_dot(13) = x(25);
  _x_dot(14) = x(26);
  _x_dot(15) = x(27);

  /**
   * Update the state vector
   * x = x + x_dot * dt
   */
  x.block(0, 0, 6, 1) += (_x_dot.block(0, 0, 6, 1) * _dt);
  x.block(6, 0, 4, 1) << _quat_new.w(), _quat_new.x(),
                         _quat_new.y(), _quat_new.z();
  x.block(10, 0, 3, 1) += (_x_dot.block(10, 0, 3, 1) * _dt);

  return x.block(0, 0, N_STATES, 1);
}

Eigen::Matrix<Quadrotor::TYPE, Quadrotor::N_MEASUREMENTS, 1> Quadrotor::measurementFunction(
    const Eigen::Matrix<Quadrotor::TYPE, Quadrotor::N_AUGMENTED_UPDATE_STATE, 1> &x)
{
  Eigen::Matrix<TYPE, N_MEASUREMENTS, 1> z; //!< The predicted measurement

  /**
   * The state x consists of 15 elements (x, y, z, vx, vy, vz, qw, qx, qy, qz,
   * ax_bias, ay_bias, az_bias, wx_bias, wy_bias, wz_bias) and 9 noise
   * elements (position_noise, velocity_noise, orientation_noise) each as a
   * 3x1 vector The measurement z consists of 9 elements the position (x, y,
   * z) and velocities (vx, vy, vz) and orientation (phi, theta, psi).
   */
  z(0) = x(0) - x(16); //!< measured x position = x - x_noise
  z(1) = x(1) - x(17); //!< measured y position = y - y_noise
  z(2) = x(2) - x(18); //!< measured z position = z - z_noise

  /**
   * The orientation is a direct linear function of the state
   */
  z(3) = x(6) - x(19); //!< measured quaternion w = qw - qw_noise
  z(4) = x(7) - x(20); //!< measured quaternion x = qx - qx_noise
  z(5) = x(8) - x(21); //!< measured quaternion y = qy - qy_noise
  z(6) = x(9) - x(22); //!< measured quaternion z = qz - qz_noise

  z.block(3, 0, 4, 1).normalize();
  return z;
}

template class UnscentedKalmanFilter<Quadrotor, Quadrotor::TYPE, Quadrotor::N_STATES,
                            Quadrotor::N_CONTROLS, Quadrotor::N_MEASUREMENTS,
                            Quadrotor::N_PROCESS_NOISES,
                            Quadrotor::N_MEASUREMENT_NOISES,
                            Quadrotor::N_HISTORY>;
