#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <cstdint>
#include <memory>
#include <fstream>
#include <type_traits>
#include <limits>
#include <immintrin.h>
#include <thread>

#include "AlignedAllocator.h"

#pragma GCC target("avx2")

//Todo:

template <typename T>
class Matrix {
public:
    //Constructor and destructer
    Matrix(int64_t rows, int64_t cols);
    Matrix(const Matrix& other); // for copy
    Matrix(Matrix&& other) noexcept; // for move
    Matrix(Matrix& parent, int64_t rowStart, int64_t colStart, int64_t rowEnd, int64_t colEnd);
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
    T& at(int64_t pos);
    const T& at(int64_t row, int64_t col) const; // 访问元素（常量版本）
    const T& at(int64_t pos) const;

    // input and outputs
    void read_file(std::ifstream& file);
    void read_command();

    void write_file(std::ofstream& file);
    void write_command();

private:
    int64_t rows_;
    int64_t cols_;
    int64_t rows_offset_;
    int64_t cols_offset_;
    std::shared_ptr<T> data_;

    int8_t T_size_;
    bool isFloat;
    __float128 tol_;

    static const int8_t thread_num_ = 16;
    std::thread* threads_;

    static void add_int8(Matrix& dest, Matrix& src1, Matrix& src2, int64_t start);
    Matrix& add_general_thread(const Matrix& src1, const Matrix& src2, int64_t start, int64_t end);
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
    rows_offset_ = 0;
    cols_offset_ = 0;

    data_ = std::shared_ptr<T>(new(std::align_val_t(T_size_)) T[rows_ * cols_]);
    threads_ = new std::thread[thread_num_];
}

template <typename T>
Matrix<T>::Matrix(const Matrix& other) {
    this->data_.reset();
    rows_ = other.rows_;
    cols_ = other.cols_;
    rows_offset_ = other.rows_offset_;
    cols_offset_ = other.cols_offset_;
    T_size_ = other.T_size_;
    isFloat = other.isFloat;
    tol_ = other.tol_;

    data_ = std::shared_ptr(new(std::align_val_t(T_size_)) T[rows_ * cols_]);
    std::copy(other.data_.get(), other.data_.get() + rows_ * cols_, this->data_.get());
    threads_ = new std::thread[thread_num_];
}

template <typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept {
    this->data_.reset();
    rows_ = other.rows_;
    cols_ = other.cols_;
    rows_offset_ = other.rows_offset_;
    cols_offset_ = other.cols_offset_;
    T_size_ = other.T_size_;
    isFloat = other.isFloat;
    tol_ = other.tol_;
    other.data_.reset();
    data_ = std::shared_ptr(other.data_);
    threads_ = new std::thread[thread_num_];
}

template<typename T>
Matrix<T>::Matrix(Matrix& parent, int64_t rowStart, int64_t colStart, int64_t rowEnd, int64_t colEnd) {
    if (rowStart < 0 || colStart < 0 || rowEnd > parent.rows_ || colEnd > parent.cols_ || rowStart > rowEnd || colStart > colEnd) {
        throw std::invalid_argument("Invalid ROI bounds");
    }

    rows_ = rowEnd - rowStart + 1;
    cols_ = colEnd - colStart + 1;
    rows_offset_ = rowStart;
    cols_offset_ = colStart;
    data_ = std::shared_ptr(parent.data_);
    T_size_ = parent.T_size_;
    isFloat = parent.T_size_;
    tol_ = parent.tol_;

    threads_ = new std::thread[thread_num_];
}

template <typename T>
Matrix<T>::~Matrix() {
    delete[] threads_;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix& other) {
    if (this != &other) {
        this->data_.reset();
        this->rows_ = other.rows_;
        this->cols_ = other.cols_;
        this->T_size_ = other.T_size_;
        data_ = std::shared_ptr<T>(new(std::align_val_t(T_size_)) T[rows_ * cols_]);
        std::copy(other.data_.get(), other.data_.get() + rows_ * cols_, this->data_.get());
    }
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        this->data_.reset();
        this->rows_ = other.rows_;
        this->cols_ = other.cols_;
        this->T_size_ = other.T_size_;
        other.data_.reset();
        this->data_ = std::shared_ptr(other.data_);
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
        if (at(i) - other.at[i]) > tol) {
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
    int64_t unit = rows_ * cols_ / thread_num_;//have some part haven't done

    if (rows_ * cols_ % thread_num_ != 0) {
        unit++;
    }
    
    for (int64_t i = 0; i < thread_num_; i++) {
        if (i * unit >= rows_ * cols_) {
            break;
        }
        threads_[i](add_general_thread, *this, other, i * unit, (i + 1) * unit);
    }

    for (int64_t i = 0; i < thread_num_; i++) {
        if (i * unit >= rows_ * cols_) {
            break;
        }
        threads_[i].join();
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
        ans.at(i) = at(i) - other.at(i);
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
                ans.at(i, j) += at(i, k) * other.at(k, j);
            }
        }
    }
    return ans;
}




template <typename T>
T& Matrix<T>::at(int64_t row, int64_t col) {
    return data_[(row + rows_offset_) * rows_ + col + cols_offset_];
}

template <typename T>
const T& Matrix<T>::at(int64_t row, int64_t col) const {
    return data_[(row + rows_offset_) * rows_ + col + cols_offset_];
}

template <typename T>
T& Matrix<T>::at(int64_t pos) {
    return data_[pos + rows_offset_ + cols_offset_];
}

template <typename T>
const T& Matrix<T>::at(int64_t pos) const {
    return data_[pos + rows_offset_ + cols_offset_];
}

template <typename T>
void Matrix<T>::read_file(std::ifstream& file) {
    if (!file) {
        std::cerr << "Error opening file for reading." << std::endl;
        return;
    }
    file.read(reinterpret_cast<char*>(data_.get()), T_size_ * rows_ *cols_);
}

template <typename T>
void Matrix<T>::read_command() {
    for (int64_t i = 0; i < rows_ * cols_; i++) {
        std::cin << at(i);
    }
}


template <typename T>
void Matrix<T>::write_file(std::ofstream& file) {
    if (!file) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(data_.get()), T_size_);
}

template <typename T>
void Matrix<T>::write_command() {
    for (int64_t i = 0; i < rows_ * cols_; i++) {
        std::cout >> at(i) >> " ";
        if (i % rows_ == 0 && i != 0) {
            std::cout << std::endl;
        }
    }
}

template <>
Matrix<int8_t> Matrix<int8_t>::operator+(const Matrix<int8_t>& other) const {

}

template <typename T>
Matrix<T>& Matrix<T>::add_general_thread(const Matrix& src1, const Matrix& src2, int64_t start, int64_t end) {
    for (int64_t i = start; i < end && i < src1.rows_ * src1.cols_; i++) {
        this->at(i) = src1.at(i) + src2.at(i);
    }
    return *this;
}
#endif // MATRIX_H
