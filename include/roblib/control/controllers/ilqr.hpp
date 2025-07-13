#pragma once

#include <Eigen/Dense>
#include <vector>
#include <iostream>

/**
 * @brief A generic model structure that the iLQR solver will use.
 *
 * The user must provide a model that conforms to this structure. It should define
 * the state and control vector types and implement the dynamics and cost functions.
 *
 * @tparam D_TYPE The data type for calculations (e.g., float, double).
 * @tparam STATE_SIZE The dimension of the state vector.
 * @tparam CONTROL_SIZE The dimension of the control vector.
 */
template <typename D_TYPE, int STATE_SIZE, int CONTROL_SIZE>
struct Model
{
  // Define core types for states and controls
  using StateVector = Eigen::Matrix<D_TYPE, STATE_SIZE, 1>;
  using ControlVector = Eigen::Matrix<D_TYPE, CONTROL_SIZE, 1>;

  // Define matrix types for linearized dynamics
  using StateMatrix = Eigen::Matrix<D_TYPE, STATE_SIZE, STATE_SIZE>;
  using ControlMatrix = Eigen::Matrix<D_TYPE, STATE_SIZE, CONTROL_SIZE>;

  /**
   * @brief The nonlinear state transition function (dynamics).
   * @param state The current state vector.
   * @param control The current control vector.
   * @param dt The time step.
   * @return The next state vector.
   */
  virtual StateVector stateTransitionFunction(const StateVector &state, const ControlVector &control, D_TYPE dt) const = 0;

  /**
   * @brief Linearizes the dynamics around a given state and control.
   * @param state The state vector to linearize around.
   * @param control The control vector to linearize around.
   * @param dt The time step.
   * @return A pair of Jacobian matrices (A, B), where A is df/dx and B is df/du.
   */
  virtual std::pair<StateMatrix, ControlMatrix> linearizeDynamics(const StateVector &state, const ControlVector &control, D_TYPE dt) const = 0;
};

/**
 * @brief A templated iterative Linear-Quadratic Regulator (iLQR) solver.
 *
 * @tparam SYSTEM_MODEL The user-defined system model class/struct.
 * @tparam D_TYPE The data type for calculations (e.g., float, double).
 * @tparam STATE_SIZE The dimension of the state vector.
 * @tparam CONTROL_SIZE The dimension of the control vector.
 * @tparam HORIZON The number of time steps in the trajectory.
 */
template <class SYSTEM_MODEL, typename D_TYPE, int STATE_SIZE, int CONTROL_SIZE, int HORIZON>
class iLQR
{
public:
    // Expose core types for convenience
  using StateVector = Eigen::Matrix<D_TYPE, STATE_SIZE, 1>;
  using ControlVector = Eigen::Matrix<D_TYPE, CONTROL_SIZE, 1>;

  using StateMatrix = Eigen::Matrix<D_TYPE, STATE_SIZE, STATE_SIZE>;
  using ControlMatrix = Eigen::Matrix<D_TYPE, STATE_SIZE, CONTROL_SIZE>;
  using FeedbackGainMatrix = Eigen::Matrix<D_TYPE, CONTROL_SIZE, STATE_SIZE>;

  using CONTROL_COST_MATRIX = Eigen::Matrix<D_TYPE, CONTROL_SIZE, CONTROL_SIZE>;

  // Trajectory types
  using StateTrajectory = std::vector<StateVector>;
  using ControlTrajectory = std::vector<ControlVector>;

  /**
   * @brief Constructor for the iLQR solver.
   */
  iLQR()
  {
    _Q.setIdentity();
    _R.setIdentity();
    _Q_final.setIdentity();

    _state_trajectory.resize(HORIZON + 1, StateVector::Zero());
    _control_trajectory.resize(HORIZON, ControlVector::Zero());
  }

  /**
   * @brief Sets the cost function matrices.
   * @param Q The running state cost matrix (penalizes state deviation).
   * @param R The running control cost matrix (penalizes control effort).
   * @param Q_final The final state cost matrix (penalizes final state deviation).
   */
  void setCost(const StateMatrix &Q, const CONTROL_COST_MATRIX &R, const StateMatrix &Q_final)
  {
    _Q = Q;
    _R = R;
    _Q_final = Q_final;
  }

  void setDt(D_TYPE dt)
  {
    _dt = dt;
  }

  /**
   * @brief Sets the goal state for the trajectory.
   * @param goal_state The desired final state.
   */
  void setGoal(const StateVector &goal_state)
  {
    _goal_state = goal_state;
  }

  /**
   * @brief Runs the iLQR optimization algorithm.
   *
   * @param x0 The initial state of the system.
   * @param max_iterations The maximum number of iLQR iterations to perform.
   * @param tolerance The convergence tolerance for the change in cost.
   * @return A pair containing the optimized state and control trajectories.
   */
  std::pair<StateTrajectory, ControlTrajectory> run(const StateVector &x0, D_TYPE dt=0, int max_iterations = 1000, D_TYPE tolerance = 1e-4, int max_line_search_attempts = 10)
  {
    if(dt > 0)
    {
      _dt = dt;
    }
    // Initialize with a zero control sequence
    _control_trajectory.assign(HORIZON, ControlVector::Zero());
    _state_trajectory[0] = x0;
    forwardPass(x0, _control_trajectory);
    D_TYPE last_cost = computeTotalCost(_state_trajectory, _control_trajectory);
    
    std::cout << "Initial Cost: " << last_cost << std::endl;

    for (int i = 0; i < max_iterations; ++i)
    {
      // 1. Backward Pass: Compute feedback gains
      auto [K_trj, k_trj] = backwardPass(_state_trajectory, _control_trajectory);

      // 2. Forward Pass with Line Search
      D_TYPE alpha = 1.0;
      bool cost_improved = false;
      for (int j = 0; j < max_line_search_attempts; ++j) // Line search attempts
      {
        auto [x_new_trj, u_new_trj] = lineSearch(_state_trajectory, _control_trajectory, K_trj, k_trj, alpha);
        D_TYPE new_cost = computeTotalCost(x_new_trj, u_new_trj);
        
        if (new_cost < last_cost)
        {
          if (std::abs(last_cost - new_cost) < tolerance)
          {
            std::cout << "Converged at iteration " << i + 1 << std::endl;
            return {x_new_trj, u_new_trj};
          }
          
          _state_trajectory = x_new_trj;
          _control_trajectory = u_new_trj;
          last_cost = new_cost;
          cost_improved = true;
          std::cout << "Iteration " << i + 1 << " | Cost: " << last_cost << " | Alpha: " << alpha << std::endl;
          break;
        }
        alpha *= 0.5; // Reduce step size
      }
      
      if (!cost_improved)
      {
        std::cout << "Failed to improve cost. Stopping." << std::endl;
        break;
      }
    }
    return {_state_trajectory, _control_trajectory};
  }

    /**
     * @brief Simulates the system forward in time with a given control sequence.
     * @param x0 The initial state.
     * @param u_trj The sequence of control inputs.
     * @return The resulting state trajectory.
     */
    StateTrajectory forwardPass(const StateVector &x0, const ControlTrajectory &u_trj)
    {
      _state_trajectory[0] = x0;
      for (int i = 0; i < HORIZON; ++i)
      {
        _state_trajectory[i + 1] = _system_model.stateTransitionFunction(_state_trajectory[i], 
                                                                         _control_trajectory[i], 
                                                                         _dt);
      }
      return _state_trajectory;
    }

    /**
     * @brief Computes the optimal feedback gains by iterating backward in time.
     * @param x_trj The current state trajectory.
     * @param u_trj The current control trajectory.
     * @return A pair of trajectories for feedback gains (K) and feedforward terms (k).
     */
    std::pair<std::vector<FeedbackGainMatrix>, ControlTrajectory> backwardPass(const StateTrajectory &x_trj, const ControlTrajectory &u_trj)
    {
      std::vector<FeedbackGainMatrix> K_trj(HORIZON);
      ControlTrajectory k_trj(HORIZON);

      StateVector z_goal_diff = x_trj.back() - _goal_state;
      StateVector p = _Q_final * z_goal_diff;
      StateMatrix P = _Q_final;

      for (int i = HORIZON - 1; i >= 0; --i)
      {
        auto [A, B] = _system_model.linearizeDynamics(x_trj[i], u_trj[i], _dt);
        
        StateVector q = _Q * (x_trj[i] - _goal_state);
        ControlVector r = _R * u_trj[i];

        // Quu, Qux, Qxx
        auto Quu = _R + B.transpose() * P * B;
        auto Qux = B.transpose() * P * A;
        
        // Invert Quu
        auto Quu_inv = Quu.inverse();

        // Feedback and feedforward gains
        K_trj[i] = -Quu_inv * Qux;
        k_trj[i] = -Quu_inv * (r + B.transpose() * p);
        
        // Update value function approximation
        p = q + A.transpose() * p + K_trj[i].transpose() * (r + B.transpose() * p) + Qux.transpose() * k_trj[i];
        P = _Q + A.transpose() * P * A + K_trj[i].transpose() * Quu * K_trj[i] + K_trj[i].transpose() * Qux + Qux.transpose() * K_trj[i];
      }
        return {K_trj, k_trj};
    }
    
    /**
     * @brief Performs a line search to find an improved trajectory.
     * @param x_trj The nominal state trajectory.
     * @param u_trj The nominal control trajectory.
     * @param K_trj The feedback gain trajectory.
     * @param k_trj The feedforward term trajectory.
     * @param alpha The line search step size.
     * @return A pair of new, potentially improved state and control trajectories.
     */
    std::pair<StateTrajectory, ControlTrajectory> lineSearch(
        const StateTrajectory &x_trj, 
        const ControlTrajectory &u_trj, 
        const std::vector<FeedbackGainMatrix> &K_trj, 
        const ControlTrajectory &k_trj, 
        D_TYPE alpha)
    {
      StateTrajectory x_new_trj(HORIZON + 1);
      ControlTrajectory u_new_trj(HORIZON);
      x_new_trj[0] = x_trj[0];

      for (int i = 0; i < HORIZON; ++i)
      {
        ControlVector delta_u = K_trj[i] * (x_new_trj[i] - x_trj[i]) + alpha * k_trj[i];
        u_new_trj[i] = u_trj[i] + delta_u;
        x_new_trj[i + 1] = _system_model.stateTransitionFunction(x_new_trj[i], u_new_trj[i], _dt);
      }
      return {x_new_trj, u_new_trj};
    }

    /**
     * @brief Computes the total cost of a given trajectory.
     * @param x_trj The state trajectory.
     * @param u_trj The control trajectory.
     * @return The total scalar cost.
     */
    D_TYPE computeTotalCost(const StateTrajectory &x_trj, const ControlTrajectory &u_trj)
    {
      D_TYPE cost = 0.0;
      for (int i = 0; i < HORIZON; ++i)
      {
        StateVector z_goal_diff = x_trj[i] - _goal_state;
        cost += 0.5 * z_goal_diff.transpose() * _Q * z_goal_diff;
        cost += 0.5 * u_trj[i].transpose() * _R * u_trj[i];
      }
      StateVector final_z_goal_diff = x_trj.back() - _goal_state;
      cost += 0.5 * final_z_goal_diff.transpose() * _Q_final * final_z_goal_diff;
      return cost;
    }

    SYSTEM_MODEL &getSystemModel()
    {
      return _system_model;
    }


private:
  // System and Algorithm Parameters
  SYSTEM_MODEL _system_model;
  D_TYPE _dt;

  // Cost function matrices
  StateMatrix _Q, _Q_final;
  Eigen::Matrix<D_TYPE, CONTROL_SIZE, CONTROL_SIZE> _R;

  // Goal state
  StateVector _goal_state;

  StateTrajectory _state_trajectory;
  ControlTrajectory _control_trajectory;
};
