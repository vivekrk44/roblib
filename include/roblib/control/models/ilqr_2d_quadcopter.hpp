#include <Eigen/Dense>

class QuadrotorModelParams
{
public:
  double mass = 0.0;
  double inertia = 0.0;
  double length = 0.0;
  double gravity = 9.81; // Default gravity in m/s^2
};

class QuadrotorModel
{
public:
  constexpr static int STATE_SIZE = 6;
  constexpr static int CONTROL_SIZE = 2;
  constexpr static int HORIZON = 200; // 2 seconds of simulation
  constexpr static double DT = 0.01;   // Simulation time step

  using StateVector   = Eigen::Matrix<double, STATE_SIZE, 1>;
  using ControlVector = Eigen::Matrix<double, CONTROL_SIZE, 1>;
  using StateMatrix   = Eigen::Matrix<double, STATE_SIZE, STATE_SIZE>;     
  using ControlMatrix = Eigen::Matrix<double, STATE_SIZE, CONTROL_SIZE>;
  
  QuadrotorModelParams _params;

  /**
   * @brief Implements the nonlinear state transition using Forward Euler integration.
   */
  StateVector stateTransitionFunction(const StateVector &state, const ControlVector &control, double dt) const
  {
    // Unpack state and control for clarity
    double px = state(0);
    double py = state(2);
    double vx = state(1);
    double vy = state(3);
    double theta = state(4);
    double omega = state(5);
    double u1 = control(0);
    double u2 = control(1);

    // Calculate accelerations
    double x_ddot = -(u1 + u2) * std::sin(theta) / _params.mass;
    double y_ddot = (u1 + u2) * std::cos(theta) / _params.mass - _params.gravity;
    double omega_dot = _params.length * (u1 - u2) / _params.inertia;

    // Integrate using Forward Euler
    StateVector next_state;
    next_state << px + vx * dt,
                  vx + x_ddot * dt,
                  py + vy * dt,
                  vy + y_ddot * dt,
                  theta + omega * dt,
                  omega + omega_dot * dt;
    return next_state;
  }

  /**
   * @brief Computes the Jacobians (A and B) of the discretized dynamics.
   */
  std::pair<StateMatrix, ControlMatrix> linearizeDynamics(const StateVector &state, const ControlVector &control, double dt) const
  {
    StateMatrix A = StateMatrix::Identity();
    ControlMatrix B = ControlMatrix::Zero();

    double theta = state(4);
    double u1 = control(0);
    double u2 = control(1);

    // Populate A matrix
    A(0, 1) = dt;
    A(1, 4) = -dt * (u1 + u2) * std::cos(theta) / _params.mass;
    A(2, 3) = dt;
    A(3, 4) = -dt * (u1 + u2) * std::sin(theta) / _params.mass;
    A(4, 5) = dt;

    // Populate B matrix
    B(1, 0) = -dt * std::sin(theta) / _params.mass;
    B(1, 1) = -dt * std::sin(theta) / _params.mass;
    B(3, 0) = dt * std::cos(theta)  / _params.mass;
    B(3, 1) = dt * std::cos(theta)  / _params.mass;
    B(5, 0) = dt * _params.length / _params.inertia;
    B(5, 1) = -dt * _params.length / _params.inertia;

    return {A, B};
  }
};
