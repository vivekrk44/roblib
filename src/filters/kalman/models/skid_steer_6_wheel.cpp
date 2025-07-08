#include <roblib/filters/models/skid_steer_6_wheels.hpp>
#include <roblib/filters/kalman/unscented_kalman_filter.hpp>
#include <roblib/utils/logger/simple_logger.hpp>

Eigen::Matrix<double, SkidSteerModel::STATE_SIZE, 1> SkidSteerModel::stateTransitionFunction(
  const Eigen::Matrix<double, AUGMENTED_PROCESS_STATE_SIZE, 1>& state,
  const Eigen::Matrix<double, CONTROL_SIZE, 1>& control,
  double dt) const
{
  Eigen::Matrix<double, STATE_SIZE, 1> new_state;
  
  // Extract states
  double px = state(0);   // x position
  double py = state(1);   // y position  
  double vx = state(2);   // body frame x velocity
  double vy = state(3);   // body frame y velocity
  double psi = state(4);  // yaw angle
  double wz = state(5);   // angular velocity (yaw rate)
  
  // Wheel RPMs (6 wheels)
  double w1 = state(6);   // Front left
  double w2 = state(7);   // Front middle
  double w3 = state(8);   // Front right
  double w4 = state(9);   // Rear left  
  double w5 = state(10);  // Rear middle
  double w6 = state(11);  // Rear right
  
  // Slip RPMs (6 wheels)
  double s1 = state(12);
  double s2 = state(13);
  double s3 = state(14);
  double s4 = state(15);
  double s5 = state(16);
  double s6 = state(17);

  // IMU biases
  double bax = state(18); // x-acceleration bias
  double bay = state(19); // y-acceleration bias  
  double bwz = state(20); // angular velocity bias
  
  // Extract control inputs (IMU measurements)
  double ax_imu = control(0);
  double ay_imu = control(1);
  double wz_imu = control(2);
  
  // Extract process noise
  double n_ax_imu = state(21);
  double n_ay_imu = state(22);
  double n_wz_imu = state(23);
  double nb_ax_imu = state(24);
  double nb_ay_imu = state(25);
  double nb_wz_imu = state(26);

  double ns_w1 = state(27);
  double ns_w2 = state(28);
  double ns_w3 = state(29);
  double ns_w4 = state(30);
  double ns_w5 = state(31);
  double ns_w6 = state(32);
  
  w1 += -(s1); // + ns_w1);
  w2 += -(s2); // + ns_w2);
  w3 += -(s3); // + ns_w3);
  w4 += -(s4); // + ns_w4);
  w5 += -(s5); // + ns_w5);
  w6 += -(s6); // + ns_w6);
  
  // Corrected IMU measurements (removing bias)
  double ax_true = ax_imu - bax - n_ax_imu;
  double ay_true = ay_imu - bay - n_ay_imu;
  double wz_true = wz_imu - bwz - n_wz_imu;
  
  // --- Kinematic Model ---
  
  // Position dynamics (global frame)
  double px_dot = vx * cos(psi) - vy * sin(psi);
  double py_dot = vx * sin(psi) + vy * cos(psi);
  
  // Average wheel speed for forward motion
  double wheel_avg = (w1 + w2 + w3 + w4 + w5 + w6) / 6.0;
  double v_desired = wheel_avg * params_.wheel_radius;
  
  // Differential wheel speeds for turning
  // Left side: w1, w4 + partial contribution from middle wheels
  // Right side: w3, w6 + partial contribution from middle wheels
  double wheel_left = (w1 + w4) / 2.0 + (w2 + w5) * 0.3;
  double wheel_right = (w3 + w6) / 2.0 + (w2 + w5) * 0.3;
  
  // Velocity dynamics (body frame) with first-order response
  double vx_dot =  ax_true;
  double vy_dot =  ay_true; 
  
  // Yaw dynamics
  double psi_dot = wz;
  double wz_dot = 0.0;
  
  // Wheel dynamics (slow drift model when not actively controlled)
  double w1_dot = vx_dot / (params_.wheel_radius);
  double w2_dot = vx_dot / (params_.wheel_radius);
  double w3_dot = vx_dot / (params_.wheel_radius);
  double w4_dot = vx_dot / (params_.wheel_radius);
  double w5_dot = vx_dot / (params_.wheel_radius);
  double w6_dot = vx_dot / (params_.wheel_radius);
  
  // Slip dynamics (return to zero slip)
  double s1_dot = ns_w1;
  double s2_dot = ns_w2;
  double s3_dot = ns_w3;
  double s4_dot = ns_w4;
  double s5_dot = ns_w5;
  double s6_dot = ns_w6;
  
  // Bias dynamics (constant bias model)
  double bax_dot = nb_ax_imu;
  double bay_dot = nb_ay_imu;
  double bwz_dot = nb_wz_imu;
  
  // Apply Euler integration with process noise
  /*new_state(0) = px_dot;*/
  /*new_state(1) = py_dot;*/
  /*new_state(2) = vx_dot;*/
  /*new_state(3) = vy_dot;*/
  /*new_state(4) = psi_dot;*/
  /*new_state(5) = wz_dot;*/
  /**/
  /*new_state(6) = w1_dot;*/
  /*new_state(7) = w2_dot;*/
  /*new_state(8) = w3_dot;*/
  /*new_state(9) = w4_dot;*/
  /*new_state(10) = w5_dot;*/
  /*new_state(11) = w6_dot;*/
  /**/
  /*new_state(12) = s1_dot;*/
  /*new_state(13) = s2_dot;*/
  /*new_state(14) = s3_dot;*/
  /*new_state(15) = s4_dot;*/
  /*new_state(16) = s5_dot;*/
  /*new_state(17) = s6_dot;*/
  /**/
  /*new_state(18) = bax_dot;*/
  /*new_state(19) = bay_dot;*/
  /*new_state(20) = bwz_dot;*/
  /**/
  /*std::stringstream ss;*/
  /*ss << "State Diff: " << new_state.transpose();*/
  /*SimpleLogger::print(ss.str(), SimpleLogger::Color::RED);*/
  /**/

  new_state(0) = px + px_dot * dt;
  new_state(1) = py + py_dot * dt;
  new_state(2) = vx + vx_dot * dt;
  new_state(3) = vy + vy_dot * dt;
  new_state(4) = psi + psi_dot * dt;
  new_state(5) = wz + wz_dot * dt;
  
  new_state(6) = w1 + w1_dot * dt;
  new_state(7) = w2 + w2_dot * dt;
  new_state(8) = w3 + w3_dot * dt;
  new_state(9) = w4 + w4_dot * dt;
  new_state(10) = w5 + w5_dot * dt;
  new_state(11) = w6 + w6_dot * dt;
  
  new_state(12) = s1 + s1_dot * dt;
  new_state(13) = s2 + s2_dot * dt;
  new_state(14) = s3 + s3_dot * dt;
  new_state(15) = s4 + s4_dot * dt;
  new_state(16) = s5 + s5_dot * dt;
  new_state(17) = s6 + s6_dot * dt;
  
  new_state(18) = bax + bax_dot * dt;
  new_state(19) = bay + bay_dot * dt;
  new_state(20) = bwz + bwz_dot * dt;
  
  // Normalize yaw angle to [-π, π]
  new_state(4) = atan2(sin(new_state(4)), cos(new_state(4)));
  
  return new_state;
}

Eigen::Matrix<double, SkidSteerModel::MEASUREMENT_SIZE, 1> SkidSteerModel::measurementFunction(
  const Eigen::Matrix<double, AUGMENTED_UPDATE_STATE_SIZE, 1>& state) const 
{
  Eigen::Matrix<double, MEASUREMENT_SIZE, 1> measurement;
  
  // Extract states
  double px = state(0);
  double py = state(1);
  double vx = state(2);
  double vy = state(3);
  double psi = state(4);
  double wz = state(5);
  double w1 = state(6);
  double w2 = state(7);
  double w3 = state(8);
  double w4 = state(9);
  double w5 = state(10);
  double w6 = state(11);
  double s1 = state(12);
  double s2 = state(13);
  double s3 = state(14);
  double s4 = state(15);
  double s5 = state(16);
  double s6 = state(17);
  
  // Extract measurement noise
  double n_px = state(21);
  double n_py = state(22);
  double n_vx = state(23);
  double n_vy = state(24);
  double n_psi = state(25);
  double n_wz = state(26);
  double n_w1 = state(27);
  double n_w2 = state(28);
  double n_w3 = state(29);
  double n_w4 = state(30);
  double n_w5 = state(31);
  double n_w6 = state(32);
  
  // Measurement model: direct observation with noise
  measurement(0) = px + n_px;     // Position x
  measurement(1) = py + n_py;     // Position y
  measurement(2) = vx + n_vx;     // Velocity x (body frame)
  measurement(3) = vy + n_vy;     // Velocity y (body frame)
  measurement(4) = psi + n_psi;   // Yaw angle
  measurement(5) = wz + n_wz;     // Angular velocity
  measurement(6) = w1 + n_w1 - s1;     // Wheel 1 RPM
  measurement(7) = w2 + n_w2 - s2;     // Wheel 2 RPM
  measurement(8) = w3 + n_w3 - s3;     // Wheel 3 RPM
  measurement(9) = w4 + n_w4 - s4;     // Wheel 4 RPM
  measurement(10) = w5 + n_w5 - s5;    // Wheel 5 RPM
  measurement(11) = w6 + n_w6 - s6;    // Wheel 6 RPM
  
  return measurement;
}

template class UnscentedKalmanFilter<SkidSteerModel, double, 
                SkidSteerModel::STATE_SIZE, 
                SkidSteerModel::CONTROL_SIZE, 
                SkidSteerModel::MEASUREMENT_SIZE, 
                SkidSteerModel::PROCESS_NOISE_SIZE, 
                SkidSteerModel::MEASUREMENT_NOISE_SIZE, 
                SkidSteerModel::HISTORY_SIZE>;
