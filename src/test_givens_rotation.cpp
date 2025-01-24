#include <Eigen/Core>
#include <iostream>
#include "utils/mathematical_tools.h"

int main()
{
    Eigen::MatrixXd random_matrix = Eigen::MatrixXd::Random(10, 10);
    std::cout << "origin matrix: \n" << random_matrix << std::endl;
    mathematical::NullSpaceProjectInplace(random_matrix, 3);
    std::cout << "After givens rotation: \n" << random_matrix << std::endl;
    return 0;
}