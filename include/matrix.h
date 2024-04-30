#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <cstdint>
#include <memory>
#include <fstream>
#include <type_traits>
#include <limits>
#include <immintrin.h>

#pragma GCC target("avx2")

//Todo:
/*
When copy and move, the elements are not totally moved
*/

template <typename T>
class Matrix {
public:
    //Constructor and destructer
    Matrix(int64_t rows, int64_t cols);
    Matrix(const Matrix& other); // for copy
    Matrix(Matrix&& other) noexcept; // for move
    ~Matrix();

    // for operator =
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;

    // for comparsion
    bool operator==(const Matrix& other) const;
    bool operator!=(const Matrix& other) const;

    // for arthimetic
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;

    // for ROI
    Matrix getROI(int64_t rowStart, int64_t colStart, int64_t rowEnd, int64_t colEnd) const;

    // others
    T& at(int64_t row, int64_t col); // look for element
    const T& at(int64_t row, int64_t col) const; // 访问元素（常量版本）

    // input and outputs
    void read_file(std::ifstream& file);
    void read_command();

    void write_file(std::ofstream& file);
    void write_command();

private:
    int64_t rows_;
    int64_t cols_;
    T* data_;

    int8_t T_size_;
    bool isFloat;
    __float128 tol_;

    
    void allocateMemory();
    void deallocateMemory();
    void copyDataFrom(const Matrix& other); // 从其他矩阵复制数据
};

template <typename T>
Matrix<T>::Matrix(int64_t rows, int64_t cols) {
    if constexpr ((std::is_same<T, int8_t>::value) || (std::is_same<T, int32_t>::value) || (std::is_same<T, int64_t>::value)) {
        isFloat = false;
    } else if constexpr ((std::is_same<T, _Float32>::value) || (std::is_same<T, _Float64>::value) || (std::is_same<T, __float128>::value)) {
        isFloat = true;
    } else {
        std::cerr << "The matrix element is not supported" << std::endl;
        return;
    }


    if constexpr (std::is_same<T, _Float32>::value) {
        tol_ = std::numeric_limits<_Float32>::epsilon();
    } else if constexpr (std::is_same<T, _Float64>::value) {
        tol_ = std::numeric_limits<_Float64>::epsilon();
    } else if constexpr (std::is_same<T, __float128>::value) {
        tol_ = std::numeric_limits<__float128>::epsilon();
    } else {
        tol_ = 0.0f;
    }


    T_size_ = sizeof(T) / 32;
    T_size_ = sizeof(T) % 32 == 0 ? T_size_ * 32 : (T_size_ + 1) * 32;
    rows_ = rows;
    cols_ = cols;

    data_ = new(std::align_val_t(T_size_)) int[rows * cols];
}

template <typename T>
Matrix<T>::Matrix(const Matrix& other) {
    delete[] this->data_;
    rows_ = other.rows_;
    cols_ = other.cols_;
    T_size_ = other.T_size_;

    data_ = new(std::align_val_t(T_size_)) int[rows_ * cols_];
    std::copy(other.data_, other.data_ + rows_ * cols_, data_);
}

template <typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept {
    delete[] this->data_;
    rows_ = other.rows_;
    cols_ = other.cols_;
    T_size_ = other.T_size_;
    data_ = other.data_;
    other.data_ = nullptr;
}

template <typename T>
Matrix<T>::~Matrix() {
    delete[] data_;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix& other) {
    if (this != &other) {
        delete[] this->data_;
        this->rows_ = other.rows_;
        this->cols_ = other.cols_;
        this->T_size_ = other.T_size_;
        this->data_ = new T[rows_ * cols_];
        std::copy(other.data_, other.data_ + rows_ * cols_, this->data_);
    }
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        delete[] this->data_;
        this->rows_ = other.rows_;
        this->cols_ = other.cols_;
        this->T_size_ = other.T_size_;
        this->data_ = other.data_;
        other.data_ = nullptr;
    }
    return *this;
}

template <typename T>
bool Matrix<T>::operator==(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        std::cerr << "Matrices with different size are compared." << std::endl;
        return false;
    }


    __float128 tol = tol_ > other.tol_ ? tol_ : other.tol_;
    tol = 3 * rows_ * cols_ * tol;
    for (int64_t i = 0; i < rows_ * cols_; i++) {
        if (abs(data_[i] - other.data_[i]) > tol) {
            return false;
        }
    }
        

    return true;
}

template <typename T>
bool Matrix<T>::operator!=(const Matrix& other) const {
    return !((*this) == other);
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        std::cerr << "Matrices with different size are added." << std::endl;
        return nullptr;
    }

    Matrix ans = Matrix(rows_, cols_);
    for (int64_t i = 0; i < rows_ * cols_; i++) {
        ans.data_[i] = data_[i] + other.data_[i];
    }

    return ans;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        std::cerr << "Matrices with different size are subbed." << std::endl;
        return nullptr;
    }

    Matrix ans = Matrix(rows_, cols_);
    for (int64_t i = 0; i < rows_ * cols_; i++) {
        ans.data_[i] = data_[i] - other.data_[i];
    }

    return ans;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        std::cerr << "Matrices with incorrect size are multiplied." << std::endl;
        return nullptr;
    }

    Matrix ans = Matrix(rows_, other.cols_);
    ans.data_ = {0};
    for (int64_t i = 0; i < rows_; i++) {
        for (int64_t j = 0; j < other.cols_; j++) {
            for (int64_t k = 0; k < cols_; k++) {
                ans.data_[i * rows_ + j] += data_[i * rows_ + k] * other.data_[k * other.rows_ + j];
            }
        }
    }

    return ans;
}


template <typename T>
void Matrix<T>::read_file(std::ifstream& file) {
    if (!file) {
        std::cerr << "Error opening file for reading." << std::endl;
        return;
    }
    file.read(reinterpret_cast<char*>(data_), T_size_ * rows_ *cols_);
}

template <typename T>
void Matrix<T>::read_command() {
    for (int64_t i = 0; i < rows_ * cols_; i++) {
        std::cin << data_[i];
    }
}


template <typename T>
void Matrix<T>::write_file(std::ofstream& file) {
    if (!file) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(data_), T_size_);
}

template <typename T>
void Matrix<T>::write_command() {
    for (int64_t i = 0; i < rows_ * cols_; i++) {
        std::cout >> data_[i] >> " ";
        if (i % rows_ == 0 && i != 0) {
            std::cout >> std::endl;
        }
    }
}
#endif // MATRIX_H
