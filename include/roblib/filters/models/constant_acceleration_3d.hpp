#include <eigen3/Eigen/Dense>

/**
 * @class ConstantVelocityModel
 * @brief A 2D constant velocity model with acceleration control inputs
 * 
 * State vector: [x, y, vx, vy]
 * Control input: [ax, ay] (accelerations)
 * Measurement: [x, y]
 */
class ConstantAccelerationModel 
{
public:
  // Define dimensions
  static const int STATE_SIZE = 16;             // [x, y, z, vx, vy, vz, qw, qx, qy, qz, bax, bay, baz, bwx, bwy, bwz]
  static const int CONTROL_SIZE = 6;            // [ax, ay]
  static const int MEASUREMENT_SIZE = 7;        // [x, y]
  static const int PROCESS_NOISE_SIZE = 12;     // Noise for each state
  static const int MEASUREMENT_NOISE_SIZE = 7;  // Noise for each measurement
  static const int HISTORY_SIZE = 200;

  
  static constexpr int AUGMENTED_PROCESS_STATE_SIZE = STATE_SIZE + PROCESS_NOISE_SIZE;    // Augmented state with Process noise
  static constexpr int AUGMENTED_UPDATE_STATE_SIZE = STATE_SIZE + MEASUREMENT_NOISE_SIZE; // Augmented state with Measurement noise

  Eigen::Vector<double, STATE_SIZE> stateTransitionFunction(
    const Eigen::Matrix<double, AUGMENTED_PROCESS_STATE_SIZE, 1>& state,
    const Eigen::Matrix<double, CONTROL_SIZE, 1>& control,
    double dt) const;
  
  Eigen::Vector<double, MEASUREMENT_SIZE> measurementFunction(
    const Eigen::Matrix<double, AUGMENTED_UPDATE_STATE_SIZE, 1>& state) const; 

  double minDt() const{return 0.0025;}

};
