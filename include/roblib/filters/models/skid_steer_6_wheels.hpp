#include <eigen3/Eigen/Dense>

/**
 * @class SkidSteerModel
 * @brief A 6-wheeled skid-steer robot model for Unscented Kalman Filter
 * 
 * State vector: [px, py, vx, vy, ψ, ωz, ω1, ω2, ω3, ω4, ω5, ω6, s1, s2, s3, s4, s5, s6, bax, bay, bωz]
 * Control input: [ax_imu, ay_imu, ωz_imu] (IMU measurements)
 * Measurement: [px, py, vx, vy, ψ, ωz, ω1, ω2, ω3, ω4, ω5, ω6]
 */
class SkidSteerModel 
{
public:
  // Define dimensions
  static const int STATE_SIZE = 21;             // [px, py, vx, vy, ψ, ωz, ω1-ω6, s1-s6, bax, bay, bωz]
  static const int CONTROL_SIZE = 3;            // [ax_imu, ay_imu, ωz_imu]
  static const int MEASUREMENT_SIZE = 12;       // [px, py, vx, vy, ψ, ωz, ω1-ω6]
  static const int PROCESS_NOISE_SIZE = 12;     // Noise for each state
  static const int MEASUREMENT_NOISE_SIZE = 12; // Noise for each measurement
  static const int HISTORY_SIZE = 200;
  
  static constexpr int AUGMENTED_PROCESS_STATE_SIZE = STATE_SIZE + PROCESS_NOISE_SIZE;    // Augmented state with Process noise
  static constexpr int AUGMENTED_UPDATE_STATE_SIZE = STATE_SIZE + MEASUREMENT_NOISE_SIZE; // Augmented state with Measurement noise

  // Robot physical parameters
  struct RobotParams {
    double wheel_base = 0.596;     // Distance between front and rear axles [m]
    double track_width = 0.477;    // Distance between left and right wheels [m] 
    double wheel_radius = 0.1;   // Wheel radius [m]
    double mass = 60.17;          // Robot mass [kg]
    double inertia = 8.57;        // Robot moment of inertia [kg⋅m²]
    
    // Time constants for dynamics
    double tau_velocity = 1e8;   // Velocity dynamics time constant [s]
    double tau_angular = 1e8;    // Angular velocity dynamics time constant [s]
    double tau_wheel = 1e8;     // Wheel dynamics time constant [s]
    double tau_slip = 1e8;       // Slip dynamics time constant [s]
    
    // Slip effect coefficient
    double slip_coefficient = 0.1;
  };

private:
  RobotParams params_;

public:
  /**
   * @brief Constructor with default parameters
   */
  SkidSteerModel() = default;
  
  /**
   * @brief Constructor with custom parameters
   */
  SkidSteerModel(const RobotParams& params) : params_(params) {}

  /**
   * @brief State transition function for skid-steer model
   * 
   * @param state Augmented state vector [21 states + 21 process noise components]
   * @param control Control input [ax_imu, ay_imu, ωz_imu]
   * @param dt Time step
   * @return New state vector [21 states]
   */
  Eigen::Matrix<double, STATE_SIZE, 1> stateTransitionFunction(
    const Eigen::Matrix<double, AUGMENTED_PROCESS_STATE_SIZE, 1>& state,
    const Eigen::Matrix<double, CONTROL_SIZE, 1>& control,
    double dt) const;

  /**
   * @brief Measurement function
   * 
   * @param state Augmented state vector [21 states + 12 measurement noise components]
   * @return Measurement vector [px, py, vx, vy, ψ, ωz, ω1-ω6]
   */
  Eigen::Matrix<double, MEASUREMENT_SIZE, 1> measurementFunction(
    const Eigen::Matrix<double, AUGMENTED_UPDATE_STATE_SIZE, 1>& state) const;

  /**
   * @brief Get/Set robot parameters
   */
  const RobotParams& getParams() const { return params_; }
  void setParams(const RobotParams& params) { params_ = params; }
};

/*namespace SkidSteerHelpers {*/
/**/
/*  Eigen::Matrix<double, SkidSteerModel::PROCESS_NOISE_SIZE, SkidSteerModel::PROCESS_NOISE_SIZE> */
/*  createProcessNoiseMatrix() {*/
/*    auto Q = Eigen::Matrix<double, SkidSteerModel::PROCESS_NOISE_SIZE, SkidSteerModel::PROCESS_NOISE_SIZE>::Zero();*/
/**/
/*    // Position process noise*/
/*    Q(0,0) = 0.001; Q(1,1) = 0.001;*/
/*    // Velocity process noise*/
/*    Q(2,2) = 0.01; Q(3,3) = 0.01;*/
/*    // Yaw and angular velocity process noise*/
/*    Q(4,4) = 0.005; Q(5,5) = 0.02;*/
/*    // Wheel RPM process noise*/
/*    for(int i = 6; i < 12; i++) Q(i,i) = 1.0;*/
/*    // Slip process noise*/
/*    for(int i = 12; i < 18; i++) Q(i,i) = 0.1;*/
/*    // Bias process noise*/
/*    Q(18,18) = 1e-6; Q(19,19) = 1e-6; Q(20,20) = 1e-6;*/
/**/
/*    return Q;*/
/*  }*/
/**/
/*  Eigen::Matrix<double, SkidSteerModel::MEASUREMENT_SIZE, SkidSteerModel::MEASUREMENT_SIZE> */
/*  createMeasurementNoiseMatrix() {*/
/*    auto R = Eigen::Matrix<double, SkidSteerModel::MEASUREMENT_SIZE, SkidSteerModel::MEASUREMENT_SIZE>::Zero();*/
/**/
/*    R(0,0) = 0.01; R(1,1) = 0.01;           // Position noise*/
/*    R(2,2) = 0.05; R(3,3) = 0.05;           // Velocity noise*/
/*    R(4,4) = 0.02;                          // Yaw noise*/
/*    R(5,5) = 0.1;                           // Angular velocity noise*/
/*    for(int i = 6; i < 12; i++) R(i,i) = 5.0; // Wheel RPM noise*/
/**/
/*    return R;*/
/*  }*/
/**/
/*  Eigen::Matrix<double, SkidSteerModel::STATE_SIZE, 1> initializeState() {*/
/*    auto x = Eigen::Matrix<double, SkidSteerModel::STATE_SIZE, 1>::Zero();*/
/**/
/*    // Initial position and orientation at origin*/
/*    x(0) = 0.0; x(1) = 0.0; x(4) = 0.0;*/
/*    // Initial velocities at zero*/
/*    x(2) = 0.0; x(3) = 0.0; x(5) = 0.0;*/
/*    // Initial wheel speeds at zero*/
/*    for(int i = 6; i < 12; i++) x(i) = 0.0;*/
/*    // Initial slip at zero*/
/*    for(int i = 12; i < 18; i++) x(i) = 0.0;*/
/*    // Initial biases at zero*/
/*    x(18) = 0.0; x(19) = 0.0; x(20) = 0.0;*/
/**/
/*    return x;*/
/*  }*/
/*}*/
