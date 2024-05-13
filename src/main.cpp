#include "matrix.h"
#include <random>
#include <cstdio>
#include <cstring>

#define min(a, b) a < b ? a : b

double* generate_arr(int size, double min, double max, std::mt19937 generator) {
    std::uniform_real_distribution<> distr_ai(min, max);
    double* ans = new double[size];
    for (int i = 0; i < size; i++) {
        ans[i] = distr_ai(generator);
    }
    return ans;
}

void test_add(double *arr1, double* arr2) {
    Matrix<double> src1(5, 4);
    Matrix<double> src2(5, 4);
    for (int i = 0; i < 20; i++) {
        src1.at(i) = arr1[i];
        src2.at(i) = arr2[i];
    }

    double max_diff = __DBL_MAX__;
    Matrix<double> ans = src1 + src2;

    for (int i = 0; i < 20; i++) {
        double x = arr1[i] + arr2[i];
        double diff = abs(x - ans.at(i));
        max_diff = min(max_diff, diff);
    }

    printf("The max_diff of adding is %lf\n", max_diff);
}

void test_minus(double *arr1, double* arr2) {
    Matrix<double> src1(5, 4);
    Matrix<double> src2(5, 4);
    for (int i = 0; i < 20; i++) {
        src1.at(i) = arr1[i];
        src2.at(i) = arr2[i];
    }

    double max_diff = __DBL_MAX__;
    Matrix<double> ans = src1 - src2;

    for (int i = 0; i < 20; i++) {
        double x = arr1[i] - arr2[i];
        double diff = abs(x - ans.at(i));
        max_diff = min(max_diff, diff);
    }

    printf("The max_diff of substracting is %lf\n", max_diff);
}

void test_mul(double *arr1, double* arr2) {
    Matrix<double> src1(5, 4);
    Matrix<double> src2(4, 5);
    for (int i = 0; i < 20; i++) {
        src1.at(i) = arr1[i];
        src2.at(i) = arr2[i];
    }

    double max_diff = __DBL_MAX__;
    Matrix<double> ans = src1 * src2;
    

    double* arr3 = new double[25];
    memset(arr3, 0.0f, 20);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 4; k++) {
                arr3[i * 5 + j] += arr1[i * 5 + k] * arr2[k * 4 + j];
            }
        }
    }

    for (int i = 0; i < 20; i++) {
        double diff = abs(arr3[i] - ans.at(i));
        max_diff = min(max_diff, diff);
    }

    delete[] arr3;
    printf("The max_diff of multiplying is %lf\n", max_diff);
}



int main() {
    //generate arr
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 eng(rd()); // Seed the generator
    double* arr1 = generate_arr(20, -1, 1, eng);
    double* arr2 = generate_arr(20, -1, 1, eng);

    test_add(arr1, arr2);
    test_minus(arr1, arr2);
    test_mul(arr1, arr2);

    delete[] arr1;
    delete[] arr2;
}