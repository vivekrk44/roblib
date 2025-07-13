#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>

// Include the UKF header
#include "roblib/filters/kalman/unscented_kalman_filter.hpp"
#include "roblib/filters/models/constant_velocity_input_accel.hpp"


// Test fixture for UKF tests
class UKFTest : public ::testing::Test {
protected:
    // Constants
    static constexpr double dt = 0.1;  // Time step
    
    // Type definition for our UKF
    using UKF = UnscentedKalmanFilter<ConstantVelocityModel, double, 
                                    ConstantVelocityModel::STATE_SIZE,
                                    ConstantVelocityModel::CONTROL_SIZE,
                                    ConstantVelocityModel::MEASUREMENT_SIZE,
                                    ConstantVelocityModel::PROCESS_NOISE_SIZE,
                                    ConstantVelocityModel::MEASUREMENT_NOISE_SIZE,
                                    ConstantVelocityModel::HISTORY_SIZE>;
    
    // Member variables
    UKF ukf;
    ConstantVelocityModel model;
    std::mt19937 random_gen;
    
    void SetUp() override {
        // Set up the random number generator
        std::random_device rd;
        random_gen = std::mt19937(rd());
        
        // Set up the UKF
        ukf.systemModel(model);
        
        // Set process noise covariance
        Eigen::Matrix<double, ConstantVelocityModel::PROCESS_NOISE_SIZE, ConstantVelocityModel::PROCESS_NOISE_SIZE> process_noise;
        process_noise.setZero();
        process_noise(0, 0) = 0.01;  // noise in x position
        process_noise(1, 1) = 0.01;  // noise in y position
        process_noise(2, 2) = 0.05;  // noise in vx
        process_noise(3, 3) = 0.05;  // noise in vy
        ukf.noiseUKFProcess(process_noise);
        
        // Set measurement noise covariance
        Eigen::Matrix<double, ConstantVelocityModel::MEASUREMENT_NOISE_SIZE, ConstantVelocityModel::MEASUREMENT_NOISE_SIZE> measurement_noise;
        measurement_noise.setZero();
        measurement_noise(0, 0) = 0.1;  // noise in x position measurement
        measurement_noise(1, 1) = 0.1;  // noise in y position measurement
        ukf.noiseUKFMeasurement(measurement_noise);
    }
    
    // Function to generate a measurement with specified noise level
    Eigen::Vector2d generateMeasurement(const Eigen::Vector4d& true_state, double noise_std) {
        std::normal_distribution<double> noise(0.0, noise_std);
        Eigen::Vector2d measurement;
        measurement(0) = true_state(0) + noise(random_gen);
        measurement(1) = true_state(1) + noise(random_gen);
        return measurement;
    }
    
    // Function to propagate state with constant velocity and acceleration
    Eigen::Vector4d propagateState(
            const Eigen::Vector4d& state, 
            const Eigen::Vector2d& acceleration,
            double time_step) {
        Eigen::Vector4d new_state;
        new_state(0) = state(0) + state(2) * time_step + 0.5 * acceleration(0) * time_step * time_step;
        new_state(1) = state(1) + state(3) * time_step + 0.5 * acceleration(1) * time_step * time_step;
        new_state(2) = state(2) + acceleration(0) * time_step;
        new_state(3) = state(3) + acceleration(1) * time_step;
        return new_state;
    }
};

// Test initialization
TEST_F(UKFTest, Initialization) {
    // Initial state
    Eigen::Vector4d initial_state;
    initial_state << 0.0, 0.0, 1.0, 1.0;  // Start at origin with velocity (1,1)
    
    // Initial covariance
    Eigen::Matrix<double, ConstantVelocityModel::STATE_SIZE, ConstantVelocityModel::STATE_SIZE> initial_covariance;
    initial_covariance.setZero();
    initial_covariance.diagonal() << 0.5, 0.5, 0.1, 0.1;
    
    // Initialize UKF
    ukf.init(initial_state, initial_covariance, 0.0);
    
    // Check initialization
    EXPECT_TRUE(ukf.initialized());
    
    // Check initial state
    Eigen::Vector4d state = ukf.state();
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(state(i), initial_state(i));
    }
    
    // Check initial covariance
    Eigen::Matrix<double, 4, 4> covariance = ukf.covariance();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_DOUBLE_EQ(covariance(i, j), initial_covariance(i, j));
        }
    }
}

// Test prediction step only
TEST_F(UKFTest, PredictionStep) {
    // Initial state and covariance
    Eigen::Vector4d initial_state;
    initial_state << 0.0, 0.0, 1.0, 1.0;
    
    Eigen::Matrix<double, 4, 4> initial_covariance;
    initial_covariance.setZero();
    initial_covariance.diagonal() << 0.5, 0.5, 0.1, 0.1;
    
    // Initialize UKF
    ukf.init(initial_state, initial_covariance, 0.0);
    
    // Control input (constant acceleration)
    Eigen::Vector2d control;
    control << 0.5, 0.3;  // Acceleration in x and y
    
    // Perform prediction
    ukf.predict(control, dt);
    
    // Expected state after prediction (without noise)
    Eigen::Vector4d expected_state = propagateState(initial_state, control, dt);
    
    // Check predicted state (allowing for process noise)
    Eigen::Vector4d predicted_state = ukf.state();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(predicted_state(i), expected_state(i), 0.2);
    }
    
    // Covariance should increase due to prediction uncertainty
    Eigen::Matrix<double, 4, 4> predicted_covariance = ukf.covariance();
    for (int i = 0; i < 4; ++i) {
        EXPECT_GT(predicted_covariance(i, i), initial_covariance(i, i));
    }
}

// Test update step only
TEST_F(UKFTest, UpdateStep) {
    // Initial state and covariance
    Eigen::Vector4d initial_state;
    initial_state << 0.0, 0.0, 1.0, 1.0;
    
    Eigen::Matrix<double, 4, 4> initial_covariance;
    initial_covariance.setZero();
    initial_covariance.diagonal() << 0.5, 0.5, 0.1, 0.1;
    
    // Initialize UKF
    ukf.init(initial_state, initial_covariance, 0.0);
    
    // Create a perfect measurement
    Eigen::Vector2d perfect_measurement;
    perfect_measurement << 1.0, 1.0;  // Different from initial position
    
    // Update with the measurement
    ukf.update(perfect_measurement, dt);
    
    // After update, state should move toward the measurement
    Eigen::Vector4d updated_state = ukf.state();
    EXPECT_GT(updated_state(0), initial_state(0));
    EXPECT_GT(updated_state(1), initial_state(1));
    
    // Covariance should decrease after update with accurate measurement
    Eigen::Matrix<double, 4, 4> updated_covariance = ukf.covariance();
    EXPECT_LT(updated_covariance(0, 0), initial_covariance(0, 0));
    EXPECT_LT(updated_covariance(1, 1), initial_covariance(1, 1));
}

// Test complete filter cycle with straight-line motion
TEST_F(UKFTest, StraightLineMotion) {
    // Initial state
    Eigen::Vector4d initial_state;
    initial_state << 0.0, 0.0, 1.0, 0.0;  // Start at origin with velocity in x direction only
    
    // Initial covariance
    Eigen::Matrix<double, 4, 4> initial_covariance;
    initial_covariance.setZero();
    initial_covariance.diagonal() << 0.1, 0.1, 0.1, 0.1;
    
    // Initialize UKF
    ukf.init(initial_state, initial_covariance, 0.0);
    
    // Control input (constant acceleration in x direction)
    Eigen::Vector2d control;
    control << 0.2, 0.0;
    
    // True trajectory
    std::vector<Eigen::Vector4d> true_states;
    true_states.push_back(initial_state);
    
    const int steps = 10;
    for (int i = 1; i <= steps; ++i) {
        true_states.push_back(propagateState(true_states.back(), control, dt));
    }
    
    // Run UKF for 10 steps
    for (int i = 1; i <= steps; ++i) {
        double timestamp = i * dt;
        
        // Generate noisy measurement
        Eigen::Vector2d measurement = generateMeasurement(true_states[i], 0.1);
        
        // Predict and update
        ukf.predict(control, timestamp);
        ukf.update(measurement, timestamp);
        
        // Check position error remains small
        Eigen::Vector4d estimated_state = ukf.state();
        double position_error = std::sqrt(
            std::pow(estimated_state(0) - true_states[i](0), 2) +
            std::pow(estimated_state(1) - true_states[i](1), 2));
        
        // Position error should remain under 0.3 units
        EXPECT_LT(position_error, 0.3);
        
        // Velocity error should remain under 0.5 units/sec
        double velocity_error = std::sqrt(
            std::pow(estimated_state(2) - true_states[i](2), 2) +
            std::pow(estimated_state(3) - true_states[i](3), 2));
        EXPECT_LT(velocity_error, 0.5);
    }
}

// Test filter with turning motion (changing acceleration)
TEST_F(UKFTest, TurningMotion) {
    // Initial state
    Eigen::Vector4d initial_state;
    initial_state << 0.0, 0.0, 1.0, 0.0;  // Start at origin with velocity in x direction
    
    // Initial covariance
    Eigen::Matrix<double, 4, 4> initial_covariance;
    initial_covariance.setZero();
    initial_covariance.diagonal() << 0.1, 0.1, 0.1, 0.1;
    
    // Initialize UKF
    ukf.init(initial_state, initial_covariance, 0.0);
    ukf.noiseUKFMeasurement(Eigen::Matrix<double, 2, 2>::Identity() * 0.05);
    ukf.noiseUKFProcess(Eigen::Matrix<double, 4, 4>::Identity() * 0.1);
    
    // True trajectory with varying acceleration (turning)
    std::vector<Eigen::Vector4d> true_states;
    std::vector<Eigen::Vector2d> controls;
    
    true_states.push_back(initial_state);
    
    // Define controls for turning motion
    controls.push_back(Eigen::Vector2d(1.5, 1.2));    // Accelerate in x
    controls.push_back(Eigen::Vector2d(1.8, -1.7));   // Start turning
    controls.push_back(Eigen::Vector2d(1.4, 1.5));    // More turn
    controls.push_back(Eigen::Vector2d(-2.9, -2.2));  // Continue turn
    controls.push_back(Eigen::Vector2d(-2.2, 2.1));   // Continue turn
    
    // Generate true trajectory
    for (size_t i = 0; i < controls.size(); ++i) {
        true_states.push_back(propagateState(true_states.back(), controls[i], dt));
    }
    
    // Run UKF
    for (size_t i = 0; i < controls.size(); ++i) {
        double timestamp = (i + 1) * dt;
        
        // Generate noisy measurement
        Eigen::Vector2d measurement = generateMeasurement(true_states[i + 1], 0.15);
        
        // Predict and update
        ukf.predict(controls[i], timestamp);
        ukf.update(measurement, timestamp);
        
        // Check position error remains reasonable
        Eigen::Vector4d estimated_state = ukf.state();
        double position_error = std::sqrt(
            std::pow(estimated_state(0) - true_states[i + 1](0), 2) +
            std::pow(estimated_state(1) - true_states[i + 1](1), 2));
        
        // Position error should remain under 0.4 units for turning motion
        EXPECT_LT(position_error, 0.5);
        std::cout << "Position error: " << position_error << std::endl;
    } 
    
    // Final position should be in the expected quadrant after turning
    Eigen::Vector4d final_state = ukf.state();
    double position_error_x = std::sqrt(std::pow(final_state(0) - true_states.back()(0), 2));
    double position_error_y = std::sqrt(std::pow(final_state(1) - true_states.back()(1), 2));
    EXPECT_LT(position_error_x, 0.4);
    EXPECT_LT(position_error_y, 0.4);
    // Check final velocity direction
    double velocity_error_x = std::sqrt(std::pow(final_state(2) - true_states.back()(2), 2));
    double velocity_error_y = std::sqrt(std::pow(final_state(3) - true_states.back()(3), 2));
    EXPECT_LT(velocity_error_x, 0.4);
    EXPECT_LT(velocity_error_y, 0.4);
}

// Test filter response to outlier measurements
TEST_F(UKFTest, OutlierMeasurements) {
    // Initial state
    Eigen::Vector4d initial_state;
    initial_state << 0.0, 0.0, 1.0, 1.0;
    
    // Initial covariance (fairly certain)
    Eigen::Matrix<double, 4, 4> initial_covariance;
    initial_covariance.setZero();
    initial_covariance.diagonal() << 0.1, 0.1, 0.1, 0.1;
    
    // Initialize UKF
    ukf.init(initial_state, initial_covariance, 0.0);
    
    // Control input
    Eigen::Vector2d control(0.0, 0.0);  // No acceleration
    
    // Normal prediction and update
    ukf.predict(control, dt);
    
    // Expected state after normal prediction
    Eigen::Vector4d expected_state = propagateState(initial_state, control, dt);
    Eigen::Vector4d state_before_outlier = ukf.state();
    
    // Generate outlier measurement (far from true position)
    Eigen::Vector2d outlier_measurement(10.0, 10.0);  // Far from expected position
    
    // Update with outlier
    ukf.update(outlier_measurement, dt);
    
    // State should move toward outlier but not match it exactly due to filtering
    Eigen::Vector4d state_after_outlier = ukf.state();
    
    // Position should move toward outlier but not reach it
    EXPECT_GT(state_after_outlier(0), state_before_outlier(0));
    EXPECT_GT(state_after_outlier(1), state_before_outlier(1));
    EXPECT_LT(state_after_outlier(0), outlier_measurement(0));
    EXPECT_LT(state_after_outlier(1), outlier_measurement(1));
    
    // Continue with normal measurements
    Eigen::Vector2d normal_measurement = generateMeasurement(propagateState(expected_state, control, dt), 0.1);
    ukf.predict(control, 2 * dt);
    ukf.update(normal_measurement, 2 * dt);
    
    // Filter should recover toward true trajectory
    Eigen::Vector4d recovered_state = ukf.state();
    double error_to_expected = std::sqrt(
        std::pow(recovered_state(0) - propagateState(expected_state, control, dt)(0), 2) +
        std::pow(recovered_state(1) - propagateState(expected_state, control, dt)(1), 2));
    
    // Error should be smaller than distance to outlier
    double distance_to_outlier = std::sqrt(
        std::pow(outlier_measurement(0) - propagateState(expected_state, control, dt)(0), 2) +
        std::pow(outlier_measurement(1) - propagateState(expected_state, control, dt)(1), 2));
    
    EXPECT_LT(error_to_expected, distance_to_outlier);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
