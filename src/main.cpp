#include <Eigen/Dense>
#include <iostream>

int main() {
    Eigen::MatrixXd m(2, 2);
    m << 1, 2,
         3, 4;

    std::cout << "Eigen is working. Here's a 2x2 matrix:\n" << m << "\n\n";
    std::cout << "Its transpose:\n" << m.transpose() << "\n";

    return 0;
}
