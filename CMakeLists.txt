cmake_minimum_required(VERSION 3.14)
project(roblib VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable generation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Option to build tests (default ON)
option(BUILD_TESTS "Build the test suite" ON)
# Create an interface library for the header-only components
add_library(roblib INTERFACE)
target_include_directories(roblib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Create a shared library
add_library(roblibModels SHARED)

# Add source files
file(GLOB_RECURSE SOURCES "src/filters/kalman/models/*cpp")
target_sources(roblibModels PRIVATE ${SOURCES})

# Add include directory
target_include_directories(roblibModels PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
# Build tests if option is enabled
if(BUILD_TESTS)
    # Enable testing
    enable_testing()
    
    # Find GTest package
    find_package(GTest REQUIRED)
    include(GoogleTest)
    
    # Create a test executable for circular buffer
    add_executable(circular_buffer_test test/datatype/circular_buffer_test.cpp)
    # Link test executable with GTest and our library
    target_link_libraries(circular_buffer_test
        PRIVATE
        GTest::GTest
        GTest::Main
        roblib
    )

    # Create a test executable for unscented kalman filter
    add_executable(unscented_kalman_filter_test test/filters/unscented_kalman_filter_test.cpp)
    # Link test executable with GTest and our library
    target_link_libraries(unscented_kalman_filter_test
        PRIVATE
        GTest::GTest
        GTest::Main
        roblib
        roblibModels
    )
    
    # Add the test to CTest
    gtest_discover_tests(circular_buffer_test)
    gtest_discover_tests(unscented_kalman_filter_test)
endif()

# Installation rules
install(TARGETS roblib
    EXPORT roblibTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install header files
install(DIRECTORY include/ 
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

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
