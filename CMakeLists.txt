cmake_minimum_required(VERSION 3.14)
project(roblib VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable generation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Define the include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create an interface library for the header-only components
add_library(roblib INTERFACE)
target_include_directories(roblib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Installation rules
install(TARGETS roblib
    EXPORT roblibTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include/roblib)

# Export the targets
install(EXPORT roblibTargets
    FILE roblibTargets.cmake
    NAMESPACE roblib::
    DESTINATION lib/cmake/roblib
)

# Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/roblibConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Configure the Config.cmake file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/roblibConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/roblibConfig.cmake"
    INSTALL_DESTINATION lib/cmake/roblib
)

# Install the Config.cmake and ConfigVersion.cmake files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/roblibConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/roblibConfigVersion.cmake"
    DESTINATION lib/cmake/roblib
)
