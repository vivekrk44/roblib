#include "roblib/datatype/circular_buffer.hpp"
#include <roblib/filters/kalman/unscented_kalman_filter.hpp>

template <class MODEL, typename D_TYPE, int STATE_SIZE, int CONTROL_SIZE, int MEASUREMENT_SIZE, int HISTORY_SIZE>
class KalmanFilter
{
public:
  using StateVector       = Eigen::Matrix<D_TYPE, STATE_SIZE, 1>;
  using ControlVector     = Eigen::Matrix<D_TYPE, CONTROL_SIZE, 1>;
  using MeasurementVector = Eigen::Matrix<D_TYPE, MEASUREMENT_SIZE, 1>;
  
  using StateMatrix       = Eigen::Matrix<D_TYPE, STATE_SIZE, STATE_SIZE>;

  using CovarianceMatrix = Eigen::Matrix<D_TYPE, STATE_SIZE, STATE_SIZE>;
  using ProcessNoise     = Eigen::Matrix<D_TYPE, STATE_SIZE, STATE_SIZE>;
  using MeasurementNoise = Eigen::Matrix<D_TYPE, MEASUREMENT_SIZE, MEASUREMENT_SIZE>;
  using InnovationMatrix = Eigen::Matrix<D_TYPE, MEASUREMENT_SIZE, MEASUREMENT_SIZE>;

  using ControlMatrix     = Eigen::Matrix<D_TYPE, STATE_SIZE,       CONTROL_SIZE>;
  using MeasurementMatrix = Eigen::Matrix<D_TYPE, MEASUREMENT_SIZE, STATE_SIZE>;
  using KalmanMatrix      = Eigen::Matrix<D_TYPE, STATE_SIZE,       MEASUREMENT_SIZE>;

  KalmanFilter ();

  /**
   * @brief Initializes the Kalman Filter with the initial state, covariance, and timestamp.
   * 
   * This function sets the initial state vector, covariance matrix, and timestamp for the Kalman Filter.
   * It also initializes the internal buffers and marks the filter as initialized.
   * 
   * @param x0 The initial state vector.
   * @param P0 The initial covariance matrix.
   * @param timestamp The initial timestamp (default is 0).
   * 
   * @note If the filter is already initialized, this function will log an error and return without reinitializing.
   */
  void init(const StateVector& x0,
            const CovarianceMatrix& P0,
            const D_TYPE timestamp = 0)
  {
    if(_initialized)
    {
      _log._ss << "KalmanFilter is already initialized.\n";
      _log.print(SimpleLogger::Color::RED);
      return;
    }

    if(_states.size > 0)
    {
      _log._ss << "KalmanFilter is already initialized.\n";
      _log.print(SimpleLogger::Color::RED);
      return;
    }
    _state = x0;
    _covariance = P0;

    _states.add(_state);
    _covariances.add(_covariance);
    _controls.add(ControlVector::Zero());
    _timestamps.add(timestamp);
    
    _initialized = true;
    _log._ss << "KalmanFilter initialized.\n";
    _log.print(SimpleLogger::Color::GREEN);
  }

  /**
   * @brief Performs the prediction step of the Kalman Filter.
   * 
   * This function predicts the next state and covariance of the system
   * based on the given control input and timestamp. It updates the internal
   * buffers with the predicted state, covariance, and timestamp.
   * 
   * @param u The control input vector.
   * @param timestamp The timestamp of the prediction step.
   * 
   * @note If the computed time difference (dt) is negative, the function will
   *       log an error and return without performing the prediction.
   */
  void predict(const ControlVector& u, D_TYPE timestamp)
  {
    D_TYPE dt = computeDt(timestamp, _timestamps.get(_states.size() - 1));
    if(dt < 0)
      return;
    _state = _states.get(_states.size() - 1);
    _covariance = _covariances.get(_covariances.size() - 1);

    predictionStep(u, dt);

    _states.add(_state);
    _covariances.add(_covariance);
    _controls.add(ControlVector::Zero());
    _timestamps.add(timestamp);
  }

  /**
   * @brief Performs the update step of the Kalman Filter.
   * 
   * This function updates the state and covariance of the system
   * based on the given measurement and timestamp. It identifies the
   * closest prior state using the timestamp, performs a prediction
   * step if necessary, and then applies the measurement update.
   * 
   * @param z The measurement vector.
   * @param timestamp The timestamp of the update step.
   * 
   * @note This function also removes outdated states and re-predicts
   *       intermediate states to maintain consistency in the buffer.
   */
  void update(const MeasurementVector& z, D_TYPE timestamp)
  {
    int index = getClosestIndex(timestamp);

    auto dt = computeDt(timestamp, _timestamps.get(index));

    _state = _states.get(index);
    _covariance = _covariances.get(index);

    predictionStep(_controls.get(index), dt);

    updateStep(z);

    _states.set(index, _state);
    _covariances.set(index, _covariance);
    _timestamps.set(index, timestamp);

    _states.removeTail(index);
    _covariances.removeTail(index);
    _controls.removeTail(index);
    _timestamps.removeTail(index);

    for(auto i = 0; i < _states.size()-1; i++)
    {
      _state = _states.get(i);
      _covariance = _covariances.get(i);
      auto control = _controls.get(i);
      auto dt = computeDt(_timestamps.get(i+1), _timestamps.get(i));
      predictionStep(control, dt);
      _states.set(i+1, _state);
      _covariances.set(i+1, _covariance);
    }
  }

  MODEL& systemModel() { return _system_model; }
private:
  
  /** 
  * @brief Finds the index of the closest timestamp that is less than or equal to the given timestamp.
  * 
  * @param timestamp The timestamp to compare against.
  * @return int The index of the closest timestamp, or -1 if no such timestamp exists.
  */
  int getClosestIndex(D_TYPE timestamp)
  {
    int index = _timestamps.size() - 1;
    for(; index >= 0 && (_timestamps.get(index) - timestamp) > 0; index--);
    return index;
  }
  /**
   * @brief Computes the time difference between two timestamps.
   * 
   * @param timestamp_next The later timestamp.
   * @param timestamp_prev The earlier timestamp.
   * @return D_TYPE The computed time difference (dt). If dt is negative, an error is logged and 0 is returned.
   */  
  D_TYPE computeDt(D_TYPE timestamp_next, D_TYPE timestamp_prev)
  {
    D_TYPE dt = timestamp_next - timestamp_prev;
    if(dt < 0)
    {
      _log._ss << "dt is negative: " << dt << "\n";
      _log._ss << "This should not happend. Check your timestamps.\n";
      _log.print(SimpleLogger::Color::RED);
      return 0;
    }
    return dt;
  }

  /**
   * @brief Performs the prediction step of the Kalman Filter.
   * 
   * This function updates the state and covariance based on the system model's
   * state transition function and its Jacobian. It also adds process noise to the covariance.
    * 
    * @param u The control input vector.
    * @param dt The time difference since the last update.
    */ 
  void predictionStep(const ControlVector& u, D_TYPE dt)
  {
    _state = _system_model.stateTransitionFunctiion(_state, u, dt);
    StateMatrix jacobian;
    jacobian = _system_model.stateTransitionMatrix(_state, u, dt);
    _covariance = jacobian * _covariance * jacobian.transpose() + _noise_process;
  }

  /**
   * @brief Performs the update step of the Kalman Filter.
   * 
   * This function computes the Kalman gain, updates the state and covariance
   * based on the measurement, and applies the innovation to the state.
   * 
   * @param z The measurement vector.
   */
  void updateStep(const MeasurementVector& z)
  {
    MeasurementVector predictedMeasurement = _system_model.measurementFunction(_state);
    MeasurementVector innovation = z - predictedMeasurement;

    MeasurementMatrix jacobian;
    jacobian = _system_model.measurementMatrix(_state);
    MeasurementMatrix innovation_covariance = jacobian * _covariance * jacobian.transpose() + _noise_measurement;

    _gain_kalman = _covariance * jacobian.transpose() * innovation_covariance.inverse();
    _covariance = _covariance - _gain_kalman * jacobian * _covariance;
    _state = _state + _gain_kalman * innovation;
  }

  MODEL _system_model; //!> The system MODEL
  //
  CircularBuffer<StateVector,       HISTORY_SIZE> _states; //!> Circular buffer for State
  CircularBuffer<ControlVector,     HISTORY_SIZE> _controls; //!> Circular buffer for Control
  CircularBuffer<CovarianceMatrix,  HISTORY_SIZE> _covariances; //!> Circular buffer for Covariance
  CircularBuffer<D_TYPE,            HISTORY_SIZE> _timestamps; //!> Circular buffer for Timestamp

  bool _initialized = false; //!> Flag to check if the filter is initialized

  D_TYPE _timestamp = 0; //!> The timestamp of the last update

  // Filter state

  StateVector _state; //!> The state vector
  CovarianceMatrix _covariance; //!> The covariance matrix 

  ProcessNoise     _noise_process; //!> The process noise Matrix
  MeasurementNoise _noise_measurement; //!> The measurement noise matrix

  KalmanMatrix _gain_kalman;

  SimpleLogger _log;

};
