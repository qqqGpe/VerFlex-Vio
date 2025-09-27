#include <Eigen/Core>
#include <iostream>
#include "mathematical_tools.h"
#include "gtest/gtest.h"

TEST(VioTest, GivensRotation)
{
    Eigen::MatrixXd random_matrix = Eigen::MatrixXd::Random(10, 10);
    std::cout << "origin matrix: \n" << random_matrix << std::endl;
    Eigen::MatrixXd R_matrix = utils::math::GivensRotation(random_matrix, 10);
    std::cout << "After givens rotation: \n" << R_matrix << std::endl;
}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}