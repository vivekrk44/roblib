#pragma once

#include <iostream>
#include <iomanip>
#include <unordered_map>
class SimpleLogger
{

public: 
  /**
   * @enum Color
   * @brief Terminal colors for logging messages
   */
  enum class Color
  {
      RED,
      GREEN,
      YELLOW,
      BLUE,
      MAGENTA,
      CYAN,
      WHITE,
      RESET
  };
  
  /**
   * @brief Log function for debugging with different colors
   * @param color Color of the message
   */
  void print(Color color = Color::WHITE)
  {
    std::cout << getColorCode(color) << _ss.str() << getColorCode(Color::RESET) << std::endl; 
    _ss.str("");
  }

  static void print(std::string strn, Color color = Color::WHITE)
  {
    std::cout << getColorCode(color) << strn << getColorCode(Color::RESET) << std::endl;
    
  }

  std::stringstream _ss;
private:
  /**
   * @brief Get the ANSI color code for the specified color
   * @param color The color enum value
   * @return The ANSI escape sequence for the color
   */
  static std::string getColorCode(Color color) 
  {
      static const std::unordered_map<Color, std::string> colorCodes = {
          {Color::RED,     "\033[1;31m"},
          {Color::GREEN,   "\033[1;32m"},
          {Color::YELLOW,  "\033[1;33m"},
          {Color::BLUE,    "\033[1;34m"},
          {Color::MAGENTA, "\033[1;35m"},
          {Color::CYAN,    "\033[1;36m"},
          {Color::WHITE,   "\033[1;37m"},
          {Color::RESET,   "\033[0m"}
      };

      auto it = colorCodes.find(color);
      return (it != colorCodes.end()) ? it->second : colorCodes.at(Color::RESET);
  }
};
