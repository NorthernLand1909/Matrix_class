#include "matrix.h"

int main() {
    Matrix<int> matrix(5, 4);
    std::ifstream file;
    file.open("output.bin", std::ios::binary);
    matrix.read_file(file);
    Matrix<int> copy(5, 4);
    copy = matrix;
}