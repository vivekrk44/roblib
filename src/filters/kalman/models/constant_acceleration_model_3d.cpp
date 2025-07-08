#include "roblib/filters/kalman/unscented_kalman_filter.hpp"
#include "roblib/filters/models/constant_acceleration_3d.hpp"

Eigen::Vector<double, ConstantAccelerationModel::STATE_SIZE> ConstantAccelerationModel::stateTransitionFunction(
  const Eigen::Matrix<double, AUGMENTED_PROCESS_STATE_SIZE, 1>& state,
  const Eigen::Matrix<double, CONTROL_SIZE, 1>& control,
  double dt) const
{

  Eigen::Vector<double, STATE_SIZE> new_state;

  // Extract states
  Eigen::Quaterniond current_orientation(state(6), state(7), state(8), state(9)); // Quaternion orientation

  // State transition: constant velocity with acceleration control
  Eigen::Vector<double, STATE_SIZE> state_dot = Eigen::Vector<double, STATE_SIZE>::Zero();
  /*state_dot(0) = state(3);  // dx/dt = vx*/
  /*state_dot(1) = state(4);  // dy/dt = velocity*/
  /*state_dot(2) = state(5);  // dz/dt = vz*/
#ifdef BODY_VEL
  state_dot.block<3, 1>(0, 0) = current_orientation * state.block<3, 1>(3, 0);
#else
  state_dot.block<3, 1>(0, 0) = current_orientation * state.block<3, 1>(3, 0);
#endif 
  Eigen::Matrix<double, 3, 1> u_w;
  u_w << (control(3) - state(13) - state(19)), (control(4) - state(14) - state(20)),
         (control(5) - state(15) - state(21));
  
  /*std::cout << "Acceleratuion: " << (control.block<3, 1>(0, 0) - state.block<3, 1>(10, 0) - state.block<3, 1>(16, 0)).transpose() << std::endl;*/
#ifdef BODY_VEL
  state_dot.block<3, 1>(3, 0) = (control.block<3, 1>(0, 0) - state.block<3, 1>(10, 0) - state.block<3, 1>(16, 0));
  Eigen::Vector3d gravity_vector(0.0, 0.0, -9.81);
  Eigen::Vector3d gravity_Vector_body = current_orientation.inverse() * gravity_vector;
  state_dot.block<3, 1>(3, 0) += gravity_Vector_body; // Add gravity in body frame
  Eigen::Vector3d coriolis_vector = -2.0 * (u_w.cross(state.block<3, 1>(3, 0))); 
  state_dot.block<3, 1>(3, 0) += coriolis_vector; // Add Coriolis effect
#else
  state_dot.block<3, 1>(3, 0) = current_orientation * (control.block<3, 1>(0, 0) - state.block<3, 1>(10, 0) - state.block<3, 1>(16, 0));
  Eigen::Vector3d gravity_vector(0.0, 0.0, -9.81);
  state_dot.block<3, 1>(3, 0) += gravity_vector; // Add gravity in body frame
#endif
  
  double norm_u_w = u_w.norm();

  Eigen::Quaterniond gyro_quaternion(state(6), state(7), state(8), state(9)); // Quaternion orientation
  double sin_norm = sin(norm_u_w * dt / 2.0);
  double cos_norm = cos(norm_u_w * dt / 2.0);
  gyro_quaternion.w() = norm_u_w < 1e-5 ? 1.0 : cos_norm;
  gyro_quaternion.x() = norm_u_w < 1e-5 ? 0.0 : u_w(0) * sin_norm / norm_u_w;
  gyro_quaternion.y() = norm_u_w < 1e-5 ? 0.0 : u_w(1) * sin_norm / norm_u_w;
  gyro_quaternion.z() = norm_u_w < 1e-5 ? 0.0 : u_w(2) * sin_norm / norm_u_w;
  
  Eigen::Quaterniond quat_new;
  quat_new = current_orientation * gyro_quaternion;
  quat_new.normalize();
  state_dot(6) = quat_new.w(); // Update quaternion w
  state_dot(7) = quat_new.x(); // Update quaternion x
  state_dot(8) = quat_new.y(); // Update quaternion y
  state_dot(9) = quat_new.z(); // Update quaternion z
  
  state_dot(10) = state(22); // Accelerometer bias x
  state_dot(11) = state(23); // Accelerometer bias y
  state_dot(12) = state(24); // Accelerometer bias z

  state_dot(13) = state(25);
  state_dot(14) = state(26);
  state_dot(15) = state(27);
  
  /*std::cout << "State dot: " << state_dot.transpose() << std::endl;*/

  new_state.block<6, 1>(0, 0) = state.block<6, 1>(0, 0) + state_dot.block<6, 1>(0, 0) * dt;
  new_state(6) = quat_new.w();
  new_state(7) = quat_new.x();
  new_state(8) = quat_new.y();
  new_state(9) = quat_new.z();
  new_state.block<6, 1>(10, 0) = state.block<6, 1>(10, 0) + state_dot.block<6, 1>(10, 0) * dt;

  return new_state;
}

 Eigen::Vector<double, ConstantAccelerationModel::MEASUREMENT_SIZE> ConstantAccelerationModel::measurementFunction(
    const Eigen::Matrix<double, AUGMENTED_UPDATE_STATE_SIZE, 1>& state) const 
  {

    Eigen::Vector<double, MEASUREMENT_SIZE> measurement;

    measurement(0) = state(0) + state(16);
    measurement(1) = state(1) + state(17);
    measurement(2) = state(2) + state(18);
    
    measurement(3) = state(6) + state(19);
    measurement(4) = state(7) + state(20);
    measurement(5) = state(8) + state(21);
    measurement(6) = state(9) + state(22);
    
    return measurement;
  }

template class UnscentedKalmanFilter<ConstantAccelerationModel, 
                double, ConstantAccelerationModel::STATE_SIZE, 
                ConstantAccelerationModel::CONTROL_SIZE, 
                ConstantAccelerationModel::MEASUREMENT_SIZE, 
                ConstantAccelerationModel::PROCESS_NOISE_SIZE, 
                ConstantAccelerationModel::MEASUREMENT_NOISE_SIZE, 
                ConstantAccelerationModel::HISTORY_SIZE>;
