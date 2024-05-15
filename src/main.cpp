#include "matrix.h"
#include <random>
#include <cstdio>

_Float64* generate_arr(int size, _Float64 min, _Float64 max) {
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 eng(rd()); // Seed the generator
    std::uniform_real_distribution<> distr_ai(min, max);
    _Float64* ans = new _Float64[size];
    for (int i = 0; i < size; i++) {
        ans[i] = distr_ai(eng);
    }
    return ans;
}

void print_ma(_Float64 *arr, int row, int col) {
    for (int i = 0; i < row * col; i++) {
        std::cout << arr[i] << " ";
        if ((i + 1) % col == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

void test_add(_Float64 *arr1, _Float64* arr2, int row, int col) {
    Matrix<_Float64> src1(row, col);
    Matrix<_Float64> src2(row, col);
    for (int i = 0; i < row * col; i++) {
        src1.at(i) = arr1[i];
        src2.at(i) = arr2[i];
    }

    _Float64 max_diff = -1;
    Matrix<_Float64> ans = src1 + src2;
    _Float64 *arr3 = new _Float64[row *col];

    for (int i = 0; i < row * col; i++) {
        arr3[i] = arr1[i] + arr2[i];
        _Float64 diff = abs(arr3[i] - ans.at(i));
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "arr3 is" << std::endl;
    print_ma(arr3, row, col);
    std::cout << "ans is" << std::endl;
    ans.write_command();

    delete[] arr3;
    std::cout << "The max_diff of adding is " << max_diff << std::endl;
}

void test_minus(_Float64 *arr1, _Float64* arr2, int row, int col) {
    Matrix<_Float64> src1(row, col);
    Matrix<_Float64> src2(row, col);
    for (int i = 0; i < row * col; i++) {
        src1.at(i) = arr1[i];
        src2.at(i) = arr2[i];
    }

    _Float64 max_diff = -1;
    Matrix<_Float64> ans = src1 - src2;
    _Float64 *arr3 = new _Float64[row * col];

    for (int i = 0; i < row * col; i++) {
        arr3[i] = arr1[i] - arr2[i];
        _Float64 diff = abs(arr3[i] - ans.at(i));
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "arr3 is" << std::endl;
    print_ma(arr3, row, col);
    std::cout << "ans is" << std::endl;
    ans.write_command();

    delete[] arr3;
    std::cout << "The max_diff of subtracting is " << max_diff << std::endl;
}

void test_mul(_Float64 *arr1, _Float64* arr2, int row1, int col1, int col2) {
    Matrix<_Float64> src1(row1, col1);
    Matrix<_Float64> src2(col1, col2);
    for (int i = 0; i < row1 * col1; i++) {
        src1.at(i) = arr1[i];
    }

    for (int i = 0; i < col1 * col2; i++) {
        src2.at(i) = arr2[i];
    }

    _Float64 max_diff = -1;
    Matrix<_Float64> ans = src1 * src2;

    _Float64* arr3 = new _Float64[row1 * col2];
    std::fill_n(arr3, row1 * col2, 0.0f);
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            for (int k = 0; k < col1; k++) {
                arr3[i * col2 + j] += arr1[i * col1 + k] * arr2[k * col2 + j];
            }
        }
    }

    for (int i = 0; i < row1 * col2; i++) {
        _Float64 diff = abs(arr3[i] - ans.at(i));
        max_diff = std::max(max_diff, diff);
    }

    // print_ma(arr3, row1, col2);
    // ans.write_command();

    std::cout << "arr3 is" << std::endl;
    print_ma(arr3, row1, col2);
    std::cout << "ans is" << std::endl;
    ans.write_command();

    delete[] arr3;

    std::cout << "The max_diff of multiplying is " << max_diff << std::endl;
}



int main() {
    //generate arr
    _Float64* arr1 = generate_arr(200000, -10, 10);
    _Float64* arr2 = generate_arr(200000, -10, 10);

    test_add(arr1, arr2, 500, 400);
    test_minus(arr1, arr2, 500, 400);
    test_mul(arr1, arr2, 400, 500, 400);

    delete[] arr1;
    delete[] arr2;
}