#include <cstdint>
#include <chrono>
#include <limits>
#include <type_traits>

template<typename SensorTimeType = uint64_t>
class SensorTimeConverter
{
private:
  static_assert(std::is_integral_v<SensorTimeType> && std::is_unsigned_v<SensorTimeType>, 
                "SensorTimeType must be an unsigned integral type");
  
  double _last_sensor_time;        // Last sensor time we processed
  double _accumulated_system_time; // Accumulated system time accounting for overflows
  double _system_start_time;       // System time when calibration occurred
  bool _initialized;
  
  static constexpr SensorTimeType MAX_SENSOR_TIME = std::numeric_limits<SensorTimeType>::max();
  static constexpr double OVERFLOW_THRESHOLD = static_cast<double>(MAX_SENSOR_TIME) + 1;

public:
  SensorTimeConverter() : _last_sensor_time(0), _accumulated_system_time(0), 
                         _system_start_time(0), _initialized(false) {}

  void initialize(double sensor_time, double system_time)
  {
    _last_sensor_time = sensor_time;
    _accumulated_system_time = 0;
    _system_start_time = system_time;
    _initialized = true;
  }

  // Convert sensor time to system time, handling overflows
  double sensorToSystemTime(double sensor_time) 
  {
    if (!_initialized) 
    {
      // Auto-calibrate on first use
      return -1;
    }

    // Detect and handle overflow
    if (sensor_time < _last_sensor_time) 
    {
      // Overflow detected - calculate how much time passed before overflow
      double time_before_overflow = (static_cast<double>(MAX_SENSOR_TIME)/1e-6) - _last_sensor_time;
      // Add the overflow period to accumulated time
      _accumulated_system_time += static_cast<double>(time_before_overflow) + 1e-6;
    }

    // Calculate elapsed time since last update
    double sensor_elapsed;
    if (sensor_time >= _last_sensor_time) 
    {
      // No overflow case
      sensor_elapsed = static_cast<double>(sensor_time - _last_sensor_time);
    } else 
    {
      // Overflow case - time since overflow occurred
      sensor_elapsed = static_cast<double>(sensor_time);
    }

    // Update tracking variables
    _last_sensor_time = sensor_time;
    _accumulated_system_time += sensor_elapsed;

    // Return system time equivalent
    return _system_start_time + _accumulated_system_time;
  }

  // Get total sensor time elapsed (accounting for overflows)
  double getTotalSensorTimeElapsed() const 
  {
    return _accumulated_system_time;
  }

  // Check if the converter has been calibrated
  bool isInitialized() const 
  {
    return _initialized;
  }

  // Reset calibration (useful if sensor restarts)
  void reset() 
  {
    _initialized = false;
    _last_sensor_time = 0;
    _accumulated_system_time = 0;
    _system_start_time = 0;
  }

  // Get information about the sensor time type
  static constexpr double getMaxSensorTime() 
  {
    return static_cast<double>(MAX_SENSOR_TIME)/1e-6;
  }

  // Get the last processed sensor time
  double getLastSensorTime() const 
  {
    return _last_sensor_time;
  }

private:
    // Your provided function to get system time
};

// Type aliases for common use cases
using SensorTimeConverter64 = SensorTimeConverter<uint64_t>;
using SensorTimeConverter32 = SensorTimeConverter<uint32_t>;
using SensorTimeConverter16 = SensorTimeConverter<uint16_t>;
using SensorTimeConverter8 = SensorTimeConverter<uint8_t>;
