#pragma once

#include <chrono>
#include "roblib/utils/logger/simple_logger.hpp"

namespace timers 
{

  enum TimerPrecision  // !< Enum for timer precision
  {
    NANOSECONDS = 1,
    MICROSECONDS = 2,
    MILLISECONDS = 4,
    SECONDS = 8
  };  // !< enum TimerPrecision

/**
 * @class TimerProfiler
 * @brief A utility class for profiling code execution time with various precision levels.
 *
 * The TimerProfiler class provides functionality to measure the execution time of code blocks
 * with different levels of precision (nanoseconds, microseconds, milliseconds, seconds).
 * It supports customizable timer names and color-coded output for debugging purposes.
 */

  class TimerProfiler 
  {
  public:
    using Color = SimpleLogger::Color;
    /**
     * @brief Constructor with a name and color.
     * @param name The name of the timer.
     * @param color The color to use for the timer output.
     */
    TimerProfiler(std::string name, Color color)
        :                                                     // !< Constructor with a name
          _name(name),                                        // !< Timer name
          _start(std::chrono::high_resolution_clock::now()),  // !< Start time
          _running(true),                                     // !< Timer running
          _precision(TimerPrecision::MILLISECONDS | TimerPrecision::MICROSECONDS),
          _color(color)
  
    {}
  
    /**
     * @brief Constructor with a name.
     * @param name The name of the timer.
     */
    TimerProfiler(std::string name)
        :                                                     // !< Constructor with a name
          _name(name),                                        // !< Timer name
          _start(std::chrono::high_resolution_clock::now()),  // !< Start time
          _running(true),                                     // !< Timer running
          _precision(TimerPrecision::MILLISECONDS | TimerPrecision::MICROSECONDS),
          _color(Color::WHITE)
  
    {}
  
    /**
     * @brief Constructor with a name and precision.
     * @param name The name of the timer.
     * @param precision The precision level for the timer.
     */
    TimerProfiler(std::string name, uint8_t precision)
        :  // !< Constructor with a name and precision
          _name(name),
          _start(std::chrono::high_resolution_clock::now()),  // !< Start time
          _running(true),                                     // !< Timer running
          _precision(precision),                              // !< Timer precision
          _color(Color::WHITE)
  
    {}
  
    /**
     * @brief Constructor with precision.
     * @param precision The precision level for the timer.
     */
    TimerProfiler(uint8_t precision)
        :  // !< Constructor with precision
          _name(std::string(__FILE__) + std::string("_") + std::to_string(__LINE__) + std::string("_") +
                std::string(__func__)),
          _start(std::chrono::high_resolution_clock::now()),  // !< Start time
          _running(true),                                     // !< Timer running
          _precision(precision),                              // !< Timer precision
          _color(Color::WHITE)
  
    {}
  
    /**
     * @brief Default constructor.
     *
     * Initializes the timer with a default name based on the file, line, and function.
     */
    TimerProfiler()
        :  // Constructor without arguments
          _name(std::string(__FILE__) + std::string("_") + std::to_string(__LINE__) + std::string("_") +
                std::string(__func__)),
          _start(std::chrono::high_resolution_clock::now()),  // !< Start time
          _running(true),                                     // !< Timer running
          _precision(TimerPrecision::MILLISECONDS),           // !< Timer precision
          _color(Color::WHITE)
    {}
  
    /**
     * @brief Destructor.
     *
     * Stops the timer if it is still running.
     */
    ~TimerProfiler()
    {
      if (_running)
        stop();
    }
  
    /**
     * @brief Set the timer precision.
     * @param precision The precision level to set.
     */
    void precision(TimerPrecision precision) { _precision = precision; }  // !< Set timer precision
  
    /**
     * @brief Get the current timer precision.
     * @return The current precision level of the timer.
     */
    uint8_t precision() const { return (uint8_t)_precision; }  // !< Get timer precision
  
    /**
     * @brief Stop the timer and log the elapsed time.
     *
     * Logs the elapsed time in the specified precision levels if profiling is enabled.
     */
    void stop() 
    {
      auto end = std::chrono::high_resolution_clock::now();
      _logger._ss << "Timer: " << _name << " ";
      if ((_precision & (uint8_t)TimerPrecision::NANOSECONDS) == TimerPrecision::NANOSECONDS)
          _logger._ss << "Nanos: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - _start).count()
                    << "ns ";
      if ((_precision & (uint8_t)TimerPrecision::MICROSECONDS) == TimerPrecision::MICROSECONDS)
          _logger._ss << "Micros: " << std::chrono::duration_cast<std::chrono::microseconds>(end - _start).count()
                    << "us ";
      if ((_precision & (uint8_t)TimerPrecision::MILLISECONDS) == TimerPrecision::MILLISECONDS)
          _logger._ss << "Millis: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - _start).count()
                    << "ms ";
      if ((_precision & (uint8_t)TimerPrecision::SECONDS) == TimerPrecision::SECONDS)
          _logger._ss << "Secnds: " << std::chrono::duration_cast<std::chrono::seconds>(end - _start).count()
                    << "s ";
      _logger.print(_color);
      _running = false;
    }

  private:
    std::string _name;                                                   // !< Timer name to print during debug
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;  // !< Start time of the timer
    bool _running;  // !< Timer running flag to check if the timer is running. Mainly used to check if the timer is
                    // stopped before destruction
    SimpleLogger _logger;
  
    uint8_t _precision;
    Color _color;
  
  };  // !< Class TimerProfiler
}  // namespace timers
