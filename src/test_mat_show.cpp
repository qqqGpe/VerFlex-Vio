#include "utils.h"
#include <Eigen/Core>


int main()
{
    Eigen::MatrixXd mat = Eigen::MatrixXd::Random(10, 10).array();
    Utils::show_eigen_matrix(mat, "test");
    return 0;
}