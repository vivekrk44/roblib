#include <gtest/gtest.h>
#include <cmath>

#include <roblib/control/controllers/ilqr.hpp>
#include <roblib/control/models/ilqr_2d_quadcopter.hpp>

typedef Eigen::Matrix<double, QuadrotorModel::STATE_SIZE, 1> StateVector;
typedef Eigen::Matrix<double, QuadrotorModel::CONTROL_SIZE, 1> ControlVector;

typedef Eigen::Matrix<double, QuadrotorModel::STATE_SIZE, QuadrotorModel::STATE_SIZE> StateMatrix;
typedef Eigen::Matrix<double, QuadrotorModel::STATE_SIZE, QuadrotorModel::CONTROL_SIZE> ControlMatrix;
typedef Eigen::Matrix<double, QuadrotorModel::CONTROL_SIZE, QuadrotorModel::CONTROL_SIZE> ControlCostMatrix;

const double DT = 0.01;

class iLQRTest : public ::testing::Test
{
protected:
    iLQR<QuadrotorModel, double, QuadrotorModel::STATE_SIZE, QuadrotorModel::CONTROL_SIZE, QuadrotorModel::HORIZON> ilqr;
};

/**
 * @brief Tests if the iLQR solver can find a trajectory to a simple goal state.
 */
TEST_F(iLQRTest, SolvesSimpleGoToGoal)
{
    QuadrotorModelParams params;
    params.mass = 2.0; // Mass of the QuadrotorModel
    params.inertia = 0.1; // Inertia of the QuadrotorModel
    params.length = 0.5; // Length of the QuadrotorModel arm
    ilqr.getSystemModel()._params = params;
    // 1. Define Cost Matrices
    StateMatrix Q, Q_final;
    ControlCostMatrix R;

    Q.setIdentity();
    Q(0,0) = 10.0; // Penalize x position error
    Q(2,2) = 10.0; // Penalize y position error

    Q_final.setIdentity();
    Q_final(0,0) = 500.0; // Heavily penalize final x position error
    Q_final(2,2) = 500.0; // Heavily penalize final y position error
    Q_final(1,1) = 50.0;  // Penalize final velocity
    Q_final(3,3) = 50.0;

    R.setIdentity();
    R *= 0.01; // Low penalty on control effort

    ilqr.setCost(Q, R, Q_final);

    // 2. Define Initial and Goal States
    StateVector x0 = StateVector::Zero();
    StateVector goal_state = StateVector::Zero();
    goal_state(0) = 5.0; // Target x = 1.0
    goal_state(2) = -5.0; // Target y = 1.0
    
    ilqr.setGoal(goal_state);

    // 3. Compute initial cost (with zero control) for comparison
    std::vector<ControlVector> initial_u_trj(QuadrotorModel::HORIZON, ControlVector::Ones());
    for (auto &u : initial_u_trj) {
        u *= 0.1; // Small initial control inputs
    }
    auto initial_x_trj = ilqr.forwardPass(x0, initial_u_trj);
    double initial_cost = ilqr.computeTotalCost(initial_x_trj, initial_u_trj);

    // 4. Run the iLQR Solver
    auto [x_trj, u_trj] = ilqr.run(x0, DT, 50, 1e-1, 10); // Run for 50 iterations
    double final_cost = ilqr.computeTotalCost(x_trj, u_trj);

    // 5. Assertions
    // Check that the optimization improved the cost
    ASSERT_LT(final_cost, initial_cost);

    // Check if the final state is reasonably close to the goal
    StateVector final_state = x_trj.back();
    EXPECT_NEAR(final_state(0), goal_state(0), 0.2); // Final x position
    EXPECT_NEAR(final_state(2), goal_state(2), 0.2); // Final y position
    std::cerr << "Final State: " << final_state.transpose() << std::endl;
    
    // Check if the robot comes to a stop
    EXPECT_NEAR(final_state(1), 0.0, 0.5); // Final x velocity
    EXPECT_NEAR(final_state(3), 0.0, 0.5); // Final y velocity
}

// --- Main function to run all tests ---
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
