#include "roblib/filters/kalman/unscented_kalman_filter.hpp"
#include "roblib/filters/models/constant_velocity_input_accel.hpp"

  Eigen::Vector4d ConstantVelocityModel::stateTransitionFunction(
    const Eigen::Matrix<double, AUGMENTED_PROCESS_STATE_SIZE, 1>& state,
    const Eigen::Matrix<double, CONTROL_SIZE, 1>& control,
    double dt) const
  {

    Eigen::Vector4d new_state;

    // Extract states
    double x = state(0);
    double y = state(1);
    double vx = state(2);
    double vy = state(3);

    // Extract control inputs (accelerations)
    double ax = control(0);
    double ay = control(1);

    // Extract process noise
    double nx = state(4);
    double ny = state(5);
    double nvx = state(6);
    double nvy = state(7);

    // State transition: constant velocity with acceleration control
    new_state(0) = x + vx * dt + 0.5 * ax * dt * dt + nx;   // x = x + vx*dt + 0.5*ax*dt² + noise
    new_state(1) = y + vy * dt + 0.5 * ay * dt * dt + ny;   // y = y + vy*dt + 0.5*ay*dt² + noise
    new_state(2) = vx + ax * dt + nvx;                      // vx = vx + ax*dt + noise
    new_state(3) = vy + ay * dt + nvy;                      // vy = vy + ay*dt + noise

    return new_state;
  }

 Eigen::Vector2d ConstantVelocityModel::measurementFunction(
    const Eigen::Matrix<double, AUGMENTED_UPDATE_STATE_SIZE, 1>& state) const 
  {

    Eigen::Vector2d measurement;

    // Extract states
    double x = state(0);
    double y = state(1);

    // Extract measurement noise
    double nx = state(4);
    double ny = state(5);

    // Measurement model: direct observation of position with noise
    measurement(0) = x + nx;  // Measured x = true x + noise
    measurement(1) = y + ny;  // Measured y = true y + noise

    return measurement;
  }

template class UnscentedKalmanFilter<ConstantVelocityModel, double, ConstantVelocityModel::STATE_SIZE, ConstantVelocityModel::CONTROL_SIZE, ConstantVelocityModel::MEASUREMENT_SIZE, ConstantVelocityModel::PROCESS_NOISE_SIZE, ConstantVelocityModel::MEASUREMENT_NOISE_SIZE, ConstantVelocityModel::HISTORY_SIZE>;
