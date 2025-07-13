#include <eigen3/Eigen/Dense>

/**
 * @class ConstantVelocityModel
 * @brief A 2D constant velocity model with acceleration control inputs
 * 
 * State vector: [x, y, vx, vy]
 * Control input: [ax, ay] (accelerations)
 * Measurement: [x, y]
 */
class ConstantVelocityModel 
{
public:
  // Define dimensions
  static const int STATE_SIZE = 4;              // [x, y, vx, vy]
  static const int CONTROL_SIZE = 2;            // [ax, ay]
  static const int MEASUREMENT_SIZE = 2;        // [x, y]
  static const int PROCESS_NOISE_SIZE = 4;      // Noise for each state
  static const int MEASUREMENT_NOISE_SIZE = 2;  // Noise for each measurement
  static const int HISTORY_SIZE = 50;

  
  static constexpr int AUGMENTED_PROCESS_STATE_SIZE = STATE_SIZE + PROCESS_NOISE_SIZE;    // Augmented state with Process noise
  static constexpr int AUGMENTED_UPDATE_STATE_SIZE = STATE_SIZE + MEASUREMENT_NOISE_SIZE; // Augmented state with Measurement noise

  /**
   * @brief State transition function for constant velocity model with acceleration control
   * 
   * @param state Augmented state vector [x, y, vx, vy, noise_x, noise_y, noise_vx, noise_vy]
   * @param control Control input [ax, ay]
   * @param dt Time step
   * @return New state vector [x, y, vx, vy]
   */
  Eigen::Vector4d stateTransitionFunction(
    const Eigen::Matrix<double, AUGMENTED_PROCESS_STATE_SIZE, 1>& state,
    const Eigen::Matrix<double, CONTROL_SIZE, 1>& control,
    double dt) const;
  /**
   * @brief Measurement function (position observations)
   * 
   * @param state Augmented state vector [x, y, vx, vy, noise_x, noise_y]
   * @return Measurement vector [x, y]
   */
  Eigen::Vector2d measurementFunction(
    const Eigen::Matrix<double, AUGMENTED_UPDATE_STATE_SIZE, 1>& state) const; 

  double minDt() const { return 1e-8; } // Minimum time step for the model
};
