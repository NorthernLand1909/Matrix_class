#include "matrix.h"
#include <random>
#include <cstdio>
#include <typeinfo>

template <typename T>
T* generate_arr(int size, T min, T max) {
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 eng(rd()); // Seed the generator
    std::uniform_real_distribution<> distr_ai(min, max);
    T* ans = new T[size];
    for (int i = 0; i < size; i++) {
        ans[i] = distr_ai(eng);
    }
    return ans;
}

template <typename T>
void print_ma(T *arr, int row, int col) {
    for (int i = 0; i < row * col; i++) {
        std::cout << (long double)arr[i] << " ";
        if ((i + 1) % col == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

template <typename T>
void test_add(T *arr1, T* arr2, int row, int col) {
    Matrix<T> src1(row, col);
    Matrix<T> src2(row, col);
    for (int i = 0; i < row * col; i++) {
        src1.at(i) = arr1[i];
        src2.at(i) = arr2[i];
    }

    T max_diff = -1;
    Matrix<T> ans = src1 + src2;
    T *arr3 = new T[row *col];

    for (int i = 0; i < row * col; i++) {
        arr3[i] = arr1[i] + arr2[i];
        T diff = abs(arr3[i] - ans.at(i));
        max_diff = std::max(max_diff, diff);
    }

    // std::cout << "arr3 is" << std::endl;
    // print_ma(arr3, row, col);
    // std::cout << "ans is" << std::endl;
    // ans.write_command();

    Matrix<T> ma(arr3, row, col, false);
    std::cout << "Is arr3 equals ans ? " << (ma == ans) << std::endl;
    delete[] arr3;
    std::cout << "The max_diff of adding is " << (long double)max_diff << std::endl;
}

template <typename T>
void test_minus(T *arr1, T* arr2, int row, int col) {
    Matrix<T> src1(row, col);
    Matrix<T> src2(row, col);
    for (int i = 0; i < row * col; i++) {
        src1.at(i) = arr1[i];
        src2.at(i) = arr2[i];
    }

    T max_diff = -1;
    Matrix<T> ans = src1 - src2;
    T *arr3 = new T[row * col];

    for (int i = 0; i < row * col; i++) {
        arr3[i] = arr1[i] - arr2[i];
        T diff = abs(arr3[i] - ans.at(i));
        max_diff = std::max(max_diff, diff);
    }

    // std::cout << "arr3 is" << std::endl;
    // print_ma(arr3, row, col);
    // std::cout << "ans is" << std::endl;
    // ans.write_command();

    Matrix<T> ma(arr3, row, col, false);
    std::cout << "Is arr3 equals ans ? " << (ma == ans) << std::endl;
    delete[] arr3;
    std::cout << "The max_diff of subtracting is " << (long double)max_diff << std::endl;
}

template <typename T>
void test_mul(T *arr1, T* arr2, int row1, int col1, int col2) {
    Matrix<T> src1(row1, col1);
    Matrix<T> src2(col1, col2);
    for (int i = 0; i < row1 * col1; i++) {
        src1.at(i) = arr1[i];
    }

    for (int i = 0; i < col1 * col2; i++) {
        src2.at(i) = arr2[i];
    }

    T max_diff = -1;
    Matrix<T> ans = src1 * src2;

    T* arr3 = new T[row1 * col2];
    std::fill_n(arr3, row1 * col2, 0.0f);
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            for (int k = 0; k < col1; k++) {
                arr3[i * col2 + j] += arr1[i * col1 + k] * arr2[k * col2 + j];
            }
        }
    }

    for (int i = 0; i < row1 * col2; i++) {
        T diff = abs(arr3[i] - ans.at(i));
        max_diff = std::max(max_diff, diff);
    }

    // std::cout << "arr3 is" << std::endl;
    // print_ma(arr3, row1, col2);
    // std::cout << "ans is" << std::endl;
    // ans.write_command();

    Matrix<T> ma(arr3, row1, col2, false);
    std::cout << "Is arr3 equals ans ? " << (ma == ans) << std::endl;
    delete[] arr3;

    std::cout << "The max_diff of multiplying is " << (long double)max_diff << std::endl;
}

template <typename T>
void test_all() {
    //generate arr
    T* arr1 = generate_arr<T>(200000, -10, 10);
    T* arr2 = generate_arr<T>(200000, -10, 10);

    test_add<T>(arr1, arr2, 500, 400);
    test_minus<T>(arr1, arr2, 500, 400);
    test_mul<T>(arr1, arr2, 400, 500, 400);

    delete[] arr1;
    delete[] arr2;
}

int main() {
    std::cout << "test for int8" << std::endl;
    test_all<int8_t>();
    std::cout << "test for int32" << std::endl;
    test_all<int32_t>();
    std::cout << "test for int64" << std::endl;
    test_all<int64_t>();
    std::cout << "test for float32" << std::endl;
    test_all<_Float32>();
    std::cout << "test for float64" << std::endl;
    test_all<_Float64>();
}