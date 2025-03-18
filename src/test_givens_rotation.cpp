#include <Eigen/Core>
#include <iostream>
#include "utils/mathematical_tools.h"

int main()
{
    Eigen::MatrixXd random_matrix = Eigen::MatrixXd::Random(10, 10);
    std::cout << "origin matrix: \n" << random_matrix << std::endl;
    // MathUtils::NullSpaceProjectInplace(random_matrix, 3);
    Eigen::MatrixXd mat_qr = MathUtils::GivensRotation(random_matrix, 10);
    std::cout << "After givens rotation: \n" << mat_qr << std::endl;
    return 0;
}