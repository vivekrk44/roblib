/* This library fuses position information from the GPS with acceleration from
 * the IMU using an Unscented Kalman Filter. The filter is implemented in the
 * UKF class from the zf_lib The comments are made in the style of the doxygen
 * documentation
 */

#include <eigen3/Eigen/Dense>

/*#define TYPE double*/
/*#define N_STATES 16*/
/*#define N_CONTROLS 6*/
/*#define N_PROCESS_NOISES 12*/
/*#define N_MEASUREMENTS 7*/
/*#define N_MEASUREMENT_NOISES 7*/
/*#define N_HISTORY 200*/

/*
 * @brief This class implements a system model for an Unscented Kalman Filter
 * for a quadrotor with the following sensors
 *       - IMU
 *       - GPS
 *       - Barometer
 *       - Magnetometer
 *  The IMU provides acceleration and angular velocity measurements in the body
 * frame as control input which is used in the prediction step to compute the
 * state transition. The GPS provides position, velocity and orientation
 * information as we take the fused output of the GPS as measurement. Here, we
 *  assume the GPS position Z axis measurement is not very accurate and tends to
 * drift. We correct this using the barometer and the rangefinder.
 *
 *  We use 16 state variables which are the following:
 *  - 3 position variables (X, Y, Z)
 *  - 3 velocity variables (X, Y, Z)
 *  - 4 orientation variables (quaternion W, X, Y, Z)
 *  - 3 accel bias variables
 *  - 3 gyro bias variables
 *
 *  There are 6 control variables which are the following:
 *  - 3 acceleration variables
 *  - 3 angular velocity variables
 *
 *  There are 12 non additive noise variables in the prediction step which are
 * the following:
 *  - 3 accel noise variables
 *  - 3 gyro noise variables
 *  - 3 accel bias noise variables
 *  - 3 gyro bias noise variables
 *
 *  There are 7 measurement variables which are the following:
 *  - 3 position variables
 *  - 4 orientation variables (quaternion W, X, Y, Z)
 *
 *  There are 7 measurement noise variables in the update step which are the
 * following:
 *  - 3 position noise variables
 *  - 4 orientation noise variables (quaternion W, X, Y, Z)
 *
 *  The state transition function is given by finding the differential of the
 * state variables with respect to time. The state transition function is given
 * by x_dot = f(x,u,n) where x is the state vector, u is the control vector and
 * n is the process noise vector. The state transition function is given by
 *
 *  x_dot = [v, Rbw * (u_a - n_a - b_a) + ge3, 0.5 * Q(u_w - n_g - b_g) * Rbw,
 * n_ba, n_bw] where
 *  - v    is the velocity in the world frame, a direct linear relation to the
 * velocity state
 *  - Rbw  is the rotation from the body frame to the world frame as a
 * quaternion or the current orientation as a quaternion
 *  - Q    is a function that converts the angular velocity to a quaternion (see
 * below for details)
 *  - u_a  is the acceleration in the body frame
 *  - n_a  is the acceleration noise
 *  - b_a  is the acceleration bias
 *  - ge3  is the gravity vector in the world frame
 *  - u_w  is the angular velocity in the body frame
 *  - n_g  is the angular velocity noise
 *  - b_g  is the angular velocity bias
 *  - n_ba is the acceleration bias noise
 *  - n_bw is the angular velocity bias noise
 *
 *  Q(wx, wy, wz)
 *  {
 *    mag = sqrt(wx^2 + wy^2 + wz^2);
 *    if (mag < 1e-5)
 *    {
 *    return [1, 0, 0, 0];
 *    }
 *    else
 *    {
 *    return [cos(mag/2), sin(mag/2) * wx/mag, sin(mag/2) * wy/mag, sin(mag/2) *
 * wz/mag];
 *    }
 *  }
 *
 *  Here the noise is not additive but used in the prediction step to compute
 * the state transition and thus the non additive noise kalman filter works best
 *
 *  The measurement model is given by
 *
 *  z = h(x,v) where z is the measurement vector, x is the state vector and v is
 * the measurement noise vector
 *
 *  z = [p + n_p, v + n_v, phi + n_phi, theta + n_theta, psi + n_psi]
 *  where
 *  - p is the position in the world frame
 *  - n_p is the position noise
 *  - v is the velocity in the world frame
 *  - n_v is the velocity noise
 *  - phi is the roll angle
 *  - n_phi is the roll angle noise
 *  - theta is the pitch angle
 *  - n_theta is the pitch angle noise
 *  - psi is the yaw angle
 *  - n_psi is the yaw angle noise
 *
 *  However the rangefinder and the barometer use a slightly different model for
 * the position Z measurement. The rangefinder model is given by z = [p_x +
 * n_p_x, p_y + n_p_y, (p_z + n_p_z)/(cos(phi) * cos(theta)), v_x + n_v_x, v_y +
 * n_v_y, v_z + n_v_z, phi + n_phi, theta + n_theta, psi + n_psi] This is
 * different because the rangefinder measures the distance from the ground and
 * not the altitude which changes depending on the orientation of the drone
 *
 *  Similarly the barometer measures the air pressure and we use the barometric
 * formula to get the height from the air pressure. The barometer model is given
 * by height = (T0 / L) * (1 - (p / p0)^(R * L / g0)) where
 *  - T0 is the temperature at sea level
 *  - L is the temperature lapse rate
 *  - p is the air pressure
 *  - p0 is the air pressure at sea level
 *  - R is the universal gas constant
 *  - g0 is the acceleration due to gravity at sea level
 *  - height is the height from the sea level
 *
 *  The measurement model for the barometer is given by
 *  pressure = p0 * (1 - (L * height) / T0)^(g0 / (R * L))
 */

class Quadrotor {

public:
  using TYPE = double;
  static const int N_STATES = 16;            //!< Number of stateTransitionFunction
  static const int N_CONTROLS = 6;           //!< Number of control inputs
  static const int N_PROCESS_NOISES = 12;    //!< Number of process noise variables
  static const int N_MEASUREMENTS = 7;       //!< Number of measurement_source
  static const int N_MEASUREMENT_NOISES = 7; //!< Number of measurement noise variables
  static const int N_HISTORY = 200;          //!< Number of history elements

  static constexpr int N_AUGMENTED_PROCESS_STATE =
      N_STATES + N_PROCESS_NOISES; //!< Number of augmented states for the process model
  static constexpr int N_AUGMENTED_UPDATE_STATE =
      N_STATES + N_MEASUREMENT_NOISES; //!< Number of augmented states for the update model
  
  Quadrotor();

  /**
   * @brief This function computes the state trasition given the augmented state
   * matrix and control input
   *
   * @param x The state vector augmented with the noise variables
   * @param u The control input vector
   *
   * @return The predicted state given current state and control input
   */
  Eigen::Matrix<TYPE, N_STATES, 1>
  stateTransitionFunction(Eigen::Matrix<TYPE, N_AUGMENTED_PROCESS_STATE, 1> x,
                          const Eigen::Matrix<TYPE, N_CONTROLS, 1> u,
                          TYPE dt = 0);  /**
   *
   * @brief This function takes the state vector augmented by the measurement
   * noise to return the predicted measurement given the state and measurement
   * noise
   *
   * @param x The state vector augmented with the measurement noise
   * @return The predicted measurement as Eigen Matrix
   */
  Eigen::Matrix<TYPE, N_MEASUREMENTS, 1> measurementFunction(
      const Eigen::Matrix<TYPE, N_AUGMENTED_UPDATE_STATE, 1> &x); 

  /**
   * @brief Updates the timestep interval
   * @param dt The timestep interval in seconds
   */
  void dt(TYPE dt) { dt = dt; }
  /**
   * @brief Returns the timestep interval
   * @return The timestep interval in seconds
   */
  TYPE dt() const { return _dt; }

  void gravity(TYPE gravity) { _gravity = gravity; }
  TYPE gravity() const { return _gravity; }

  TYPE _dt;

private:
  TYPE _gravity = -9.81;                   //!< Gravity constant
  Eigen::Matrix<TYPE, N_STATES, 1> _x_dot; //!< State derivative

  Eigen::Quaternion<TYPE> _Rgyro_quat;   //!< Rotation matrix from body to world frame
  Eigen::Quaternion<TYPE> _quat_current; //!< Current quaternion
};
