#include "matrix.h"
#include <random>
#include <cstdio>

double* generate_arr(int size, double min, double max) {
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 eng(rd()); // Seed the generator
    std::uniform_real_distribution<> distr_ai(min, max);
    double* ans = new double[size];
    for (int i = 0; i < size; i++) {
        ans[i] = 1;
    }
    return ans;
}

void print_ma(double *arr, int row, int col) {
    for (int i = 0; i < row * col; i++) {
        printf("%lf ", arr[i]);
        if ((i + 1) % col == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

void test_add(double *arr1, double* arr2, int row, int col) {
    Matrix<double> src1(row, col);
    Matrix<double> src2(row, col);
    for (int i = 0; i < row * col; i++) {
        src1.at(i) = arr1[i];
        src2.at(i) = arr2[i];
    }

    double max_diff = __DBL_MIN__;
    Matrix<double> ans = src1 + src2;
    double *arr3 = new double[row *col];

    for (int i = 0; i < row * col; i++) {
        arr3[i] = arr1[i] + arr2[i];
        double diff = abs(arr3[i] - ans.at(i));
        max_diff = std::max(max_diff, diff);
    }
    delete[] arr3;
    printf("The max_diff of adding is %lf\n", max_diff);
}

void test_minus(double *arr1, double* arr2, int row, int col) {
    Matrix<double> src1(row, col);
    Matrix<double> src2(row, col);
    for (int i = 0; i < row * col; i++) {
        src1.at(i) = arr1[i];
        src2.at(i) = arr2[i];
    }

    double max_diff = __DBL_MIN__;
    Matrix<double> ans = src1 - src2;
    double *arr3 = new double[row * col];

    for (int i = 0; i < row * col; i++) {
        arr3[i] = arr1[i] - arr2[i];
        double diff = abs(arr3[i] - ans.at(i));
        max_diff = std::max(max_diff, diff);
    }

    delete[] arr3;
    printf("The max_diff of substracting is %lf\n", max_diff);
}

void test_mul(double *arr1, double* arr2, int row1, int col1, int col2) {
    Matrix<double> src1(row1, col1);
    Matrix<double> src2(col1, col2);
    for (int i = 0; i < row1 * col1; i++) {
        src1.at(i) = arr1[i];
    }

    for (int i = 0; i < col1 * col2; i++) {
        src2.at(i) = arr2[i];
    }

    double max_diff = __DBL_MIN__;
    Matrix<double> ans = src1 * src2;
    
    double* arr3 = new double[row1 * col2];
    std::fill_n(arr3, row1 * col2, 0.0f);
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            for (int k = 0; k < col1; k++) {
                arr3[i * col2 + j] += arr1[i * col1 + k] * arr2[k * col2 + j];
            }
        }
    }

    for (int i = 0; i < row1 * col2; i++) {
        double diff = abs(arr3[i] - ans.at(i));
        max_diff = std::max(max_diff, diff);
    }
    delete[] arr3;

    printf("The max_diff of multiplying is %lf\n", max_diff);
}



int main() {
    //generate arr
    double* arr1 = generate_arr(20, -1, 1);
    double* arr2 = generate_arr(20, -1, 1);

    test_add(arr1, arr2, 5, 4);
    test_minus(arr1, arr2, 5, 4);
    test_mul(arr1, arr2, 5, 4, 5);

    delete[] arr1;
    delete[] arr2;
}