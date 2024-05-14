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

#pragma GCC target("avx2")

//Todo: add an error when try to get elements out of range

template <typename T>
class Matrix {
public:
    //Constructor and destructer
    Matrix(int64_t rows, int64_t cols);
    Matrix(const Matrix& other); // for copy
    Matrix(Matrix&& other) noexcept; // for move
    Matrix(Matrix& parent, int64_t rowStart, int64_t colStart, int64_t rowEnd, int64_t colEnd); // for ROI
    ~Matrix();

    // for operator =
    Matrix& operator=(const Matrix& other); // for copy
    Matrix& operator=(Matrix&& other) noexcept; // for move

    // for comparsion
    // only matrices with the same size are supported!
    bool operator==(const Matrix& other) const;
    bool operator!=(const Matrix& other) const;

    // for arthimetic
    // only matrices with correct size are supported!
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;


    // others
    T& at(int64_t row, int64_t col); // look for element
    T& at(int64_t pos); // directly look for corresponding position of the array
    const T& at(int64_t row, int64_t col) const;
    const T& at(int64_t pos) const;

    // IO
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

    int8_t T_size_; // The size used for aligned memory allocated
    bool isFloat;
    __float128 tol_; // Judging for equlity

    static const int8_t thread_num_ = 16; // thread numbers

    static void thread_add(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t& start, const int64_t& end);
    static void thread_subtract(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t& start, const int64_t& end);
    static void thread_mul(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t num, const bool& isRow);
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

}

template <typename T>
Matrix<T>::~Matrix() {
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
        if (at(i) - other.at(i) > tol) {
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
        throw std::invalid_argument("Matrices with different sizes cannot be added.");
    }

    Matrix ans = Matrix(rows_, cols_);
    int64_t unit = rows_ * cols_ / thread_num_;

    if (rows_ * cols_ % thread_num_ != 0) {
        unit++;
    }
    std::thread* threads_ = new std::thread[thread_num_];
    
    for (int64_t i = 0; i < thread_num_; i++) {
        if (i * unit >= rows_ * cols_) {
            break;
        }
        threads_[i] = std::thread([&ans, this, &other, i, unit]() {
            this->thread_add(ans, *this, other, i * unit, (i + 1) * unit);
        });
    }

    for (int64_t i = 0; i < thread_num_; i++) {
        if (i * unit >= rows_ * cols_) {
            break;
        }
        if (threads_[i].joinable()) {
            threads_[i].join();
        }
    }
    delete[] threads_;

    return ans;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrices with different sizes cannot be subtracted.");
    }

    Matrix ans = Matrix(rows_, cols_);
    int64_t unit = rows_ * cols_ / thread_num_;

    if (rows_ * cols_ % thread_num_ != 0) {
        unit++;
    }
    std::thread* threads_ = new std::thread[thread_num_];
    
    for (int64_t i = 0; i < thread_num_; i++) {
        if (i * unit >= rows_ * cols_) {
            break;
        }
        threads_[i] = std::thread([&ans, this, &other, i, unit]() {
            this->thread_subtract(ans, *this, other, i * unit, (i + 1) * unit);
        });
    }

    for (int64_t i = 0; i < thread_num_; i++) {
        if (i * unit >= rows_ * cols_) {
            break;
        }
        if (threads_[i].joinable()) {
            threads_[i].join();
        }
    }
    delete[] threads_;

    return ans;
}



template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrices with incorrect size are multiplied.");
    }

    Matrix ans = Matrix(rows_, other.cols_);
    std::thread* threads_ = new std::thread[thread_num_];
    std::fill_n(ans.data_.get(), rows_ * other.cols_, static_cast<T>(0));
    for (int i = 0; i < thread_num_; i++) {
        threads_[i] = std::thread([&, this, i]() {
            this->thread_mul(ans, *this, other, i, rows_ >= other.cols_);
        });
    }


    for (int64_t i = 0; i < thread_num_; i++) {
        if (threads_[i].joinable()) {
            threads_[i].join();
        }
    }
    delete[] threads_;
    return ans;
}


template <typename T>
T& Matrix<T>::at(int64_t row, int64_t col) {
    //printf("find %ld\n", (row + rows_offset_) * cols_ + col + cols_offset_);
    if (row > rows_ || col > cols_) {
        throw std::invalid_argument("Matrix index out of bound");
    }
    return data_.get()[(row + rows_offset_) * cols_ + col + cols_offset_];
}

template <typename T>
const T& Matrix<T>::at(int64_t row, int64_t col) const {
    //printf("find %ld\n", (row + rows_offset_) * cols_ + col + cols_offset_);
    return data_.get()[(row + rows_offset_) * cols_ + col + cols_offset_];
}

template <typename T>
T& Matrix<T>::at(int64_t pos) {
    return data_.get()[pos + rows_offset_ * cols_ + cols_offset_];
}

template <typename T>
const T& Matrix<T>::at(int64_t pos) const {
    return data_.get()[pos + rows_offset_ * cols_ + cols_offset_];
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
        std::cout << at(i) << " ";
        if ((i + 1) % rows_ == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

template <typename T>
void Matrix<T>::thread_add(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t& start, const int64_t& end) {
    for (int64_t i = start; i < end && i < src1.rows_ * src1.cols_; i++) {
        dest.at(i) = src1.at(i) + src2.at(i);
    }
}

template <typename T>
void Matrix<T>::thread_subtract(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t& start, const int64_t& end) {
    for (int64_t i = start; i < end && i < src1.rows_ * src1.cols_; i++) {
        dest.at(i) = src1.at(i) - src2.at(i);
    }
}

template <typename T>
void Matrix<T>::thread_mul(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t num, const bool& isRow) {
    if (isRow) {
        if (num >= src1.rows_) {
            return;
        }
        for (int64_t i = num; i < src1.rows_; i += thread_num_) {
            for (int64_t j = 0; j < src2.cols_; j++) {
                for (int64_t k = 0; k < src1.cols_; k++) {
                    dest.at(i, j) += src1.at(i, k) * src2.at(k, j);
                }
            }
        }
    } else {
        if (num >= src2.cols_) {
            return;
        }
        for (int64_t i = num; i < src2.cols_; i += thread_num_) {
            for (int64_t j = 0; j < src1.rows_; j++) {
                for (int64_t k = 0; k < src1.cols_; k++) {
                    dest.at(i, j) += src1.at(i, k) * src2.at(k, j);
                }
            }
        }
    }
}

template <>
void Matrix<int8_t>::thread_add(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t& start, const int64_t& end) {
    int64_t num = (std::min(end, src1.rows_ * src1.cols_) - start) / 128;
    __m256i ymm[12];
    for (int64_t i = num; i < num; i++) {
        ymm[0] = _mm256_loadu_si256((__m256i*)&src1.at(start + num * 128));
        ymm[1] = _mm256_loadu_si256((__m256i*)&src1.at(start + num * 128 + 32));
        ymm[2] = _mm256_loadu_si256((__m256i*)&src1.at(start + num * 128 + 64));
        ymm[3] = _mm256_loadu_si256((__m256i*)&src1.at(start + num * 128 + 96));

        ymm[4] = _mm256_loadu_si256((__m256i*)&src2.at(start + num * 128));
        ymm[5] = _mm256_loadu_si256((__m256i*)&src2.at(start + num * 128 + 32));
        ymm[6] = _mm256_loadu_si256((__m256i*)&src2.at(start + num * 128 + 64));
        ymm[7] = _mm256_loadu_si256((__m256i*)&src2.at(start + num * 128 + 96));

        ymm[8] = _mm256_add_epi8(ymm[0], ymm[4]);
        ymm[9] = _mm256_add_epi8(ymm[1], ymm[5]);
        ymm[10] = _mm256_add_epi8(ymm[2], ymm[6]);
        ymm[11] = _mm256_add_epi8(ymm[3], ymm[7]);

        _mm256_storeu_si256((__m256i*)&dest.at(start + num * 128), ymm[8]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + num * 128 + 32), ymm[9]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + num * 128 + 64), ymm[10]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + num * 128 + 96), ymm[11]);
    }

    for (int64_t i = start + num * 128; i < end && i < src1.rows_ * src1.cols_; i++) {
        dest.at(i) = src1.at(i) + src2.at(i);
    }
}

template <>
void Matrix<int32_t>::thread_add(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t& start, const int64_t& end) {
    int64_t num = (std::min(end, src1.rows_ * src1.cols_) - start) / 32;
    __m256i ymm[12];
    for (int64_t i = num; i < num; i++) {
        ymm[0] = _mm256_loadu_si256((__m256i*)&src1.at(start + num * 32));
        ymm[1] = _mm256_loadu_si256((__m256i*)&src1.at(start + num * 32 + 8));
        ymm[2] = _mm256_loadu_si256((__m256i*)&src1.at(start + num * 32 + 16));
        ymm[3] = _mm256_loadu_si256((__m256i*)&src1.at(start + num * 32 + 24));

        ymm[4] = _mm256_loadu_si256((__m256i*)&src2.at(start + num * 32));
        ymm[5] = _mm256_loadu_si256((__m256i*)&src2.at(start + num * 32 + 8));
        ymm[6] = _mm256_loadu_si256((__m256i*)&src2.at(start + num * 32 + 16));
        ymm[7] = _mm256_loadu_si256((__m256i*)&src2.at(start + num * 32 + 24));

        ymm[8] = _mm256_add_epi32(ymm[0], ymm[4]);
        ymm[9] = _mm256_add_epi32(ymm[1], ymm[5]);
        ymm[10] = _mm256_add_epi32(ymm[2], ymm[6]);
        ymm[11] = _mm256_add_epi32(ymm[3], ymm[7]);

        _mm256_storeu_si256((__m256i*)&dest.at(start + num * 32), ymm[8]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + num * 32 + 8), ymm[9]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + num * 32 + 16), ymm[10]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + num * 32 + 24), ymm[11]);
    }

    for (int64_t i = start + num * 32; i < end && i < src1.rows_ * src1.cols_; i++) {
        dest.at(i) = src1.at(i) + src2.at(i);
    }
}

template <>
void Matrix<int64_t>::thread_add(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t& start, const int64_t& end) {
    int64_t num = (std::min(end, src1.rows_ * src1.cols_) - start) / 16;
    __m256i ymm[12];
    for (int64_t i = num; i < num; i++) {
        ymm[0] = _mm256_loadu_si256((__m256i*)&src1.at(start + num * 16));
        ymm[1] = _mm256_loadu_si256((__m256i*)&src1.at(start + num * 16 + 4));
        ymm[2] = _mm256_loadu_si256((__m256i*)&src1.at(start + num * 16 + 8));
        ymm[3] = _mm256_loadu_si256((__m256i*)&src1.at(start + num * 16 + 12));

        ymm[4] = _mm256_loadu_si256((__m256i*)&src2.at(start + num * 16));
        ymm[5] = _mm256_loadu_si256((__m256i*)&src2.at(start + num * 16 + 4));
        ymm[6] = _mm256_loadu_si256((__m256i*)&src2.at(start + num * 16 + 8));
        ymm[7] = _mm256_loadu_si256((__m256i*)&src2.at(start + num * 16 + 12));

        ymm[8] = _mm256_add_epi64(ymm[0], ymm[4]);
        ymm[9] = _mm256_add_epi64(ymm[1], ymm[5]);
        ymm[10] = _mm256_add_epi64(ymm[2], ymm[6]);
        ymm[11] = _mm256_add_epi64(ymm[3], ymm[7]);

        _mm256_storeu_si256((__m256i*)&dest.at(start + num * 16), ymm[8]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + num * 16 + 4), ymm[9]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + num * 16 + 8), ymm[10]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + num * 16 + 12), ymm[11]);
    }

    for (int64_t i = start + num * 16; i < end && i < src1.rows_ * src1.cols_; i++) {
        dest.at(i) = src1.at(i) + src2.at(i);
    }
}

template <>
void Matrix<_Float32>::thread_add(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t& start, const int64_t& end) {
    int64_t num = (std::min(end, src1.rows_ * src1.cols_) - start) / 32;
    __m256 ymm[12];
    for (int64_t i = num; i < num; i++) {
        ymm[0] = _mm256_load_ps(&src1.at(start + num * 32));
        ymm[1] = _mm256_load_ps(&src1.at(start + num * 32 + 8));
        ymm[2] = _mm256_load_ps(&src1.at(start + num * 32 + 16));
        ymm[3] = _mm256_load_ps(&src1.at(start + num * 32 + 24));

        ymm[4] = _mm256_load_ps(&src2.at(start + num * 32));
        ymm[5] = _mm256_load_ps(&src2.at(start + num * 32 + 8));
        ymm[6] = _mm256_load_ps(&src2.at(start + num * 32 + 16));
        ymm[7] = _mm256_load_ps(&src2.at(start + num * 32 + 24));

        ymm[8] = _mm256_add_ps(ymm[0], ymm[4]);
        ymm[9] = _mm256_add_ps(ymm[1], ymm[5]);
        ymm[10] = _mm256_add_ps(ymm[2], ymm[6]);
        ymm[11] = _mm256_add_ps(ymm[3], ymm[7]);

        _mm256_store_ps(&dest.at(start + num * 32), ymm[8]);
        _mm256_store_ps(&dest.at(start + num * 32 + 8), ymm[9]);
        _mm256_store_ps(&dest.at(start + num * 32 + 16), ymm[10]);
        _mm256_store_ps(&dest.at(start + num * 32 + 24), ymm[11]);
    }

    for (int64_t i = start + num * 32; i < end && i < src1.rows_ * src1.cols_; i++) {
        dest.at(i) = src1.at(i) + src2.at(i);
    }
}

template <>
void Matrix<_Float64>::thread_add(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t& start, const int64_t& end) {
    int64_t num = (std::min(end, src1.rows_ * src1.cols_) - start) / 16;
    __m256d ymm[12];
    for (int64_t i = num; i < num; i++) {
        ymm[0] = _mm256_load_pd(&src1.at(start + num * 16));
        ymm[1] = _mm256_load_pd(&src1.at(start + num * 16 + 4));
        ymm[2] = _mm256_load_pd(&src1.at(start + num * 16 + 8));
        ymm[3] = _mm256_load_pd(&src1.at(start + num * 16 + 12));

        ymm[4] = _mm256_load_pd(&src2.at(start + num * 16));
        ymm[5] = _mm256_load_pd(&src2.at(start + num * 16 + 4));
        ymm[6] = _mm256_load_pd(&src2.at(start + num * 16 + 8));
        ymm[7] = _mm256_load_pd(&src2.at(start + num * 16 + 12));

        ymm[8] = _mm256_add_pd(ymm[0], ymm[4]);
        ymm[9] = _mm256_add_pd(ymm[1], ymm[5]);
        ymm[10] = _mm256_add_pd(ymm[2], ymm[6]);
        ymm[11] = _mm256_add_pd(ymm[3], ymm[7]);

        _mm256_store_pd(&dest.at(start + num * 16), ymm[8]);
        _mm256_store_pd(&dest.at(start + num * 16 + 4), ymm[9]);
        _mm256_store_pd(&dest.at(start + num * 16 + 8), ymm[10]);
        _mm256_store_pd(&dest.at(start + num * 16 + 12), ymm[11]);
    }

    for (int64_t i = start + num * 16; i < end && i < src1.rows_ * src1.cols_; i++) {
        dest.at(i) = src1.at(i) + src2.at(i);
    }
}

#endif // MATRIX_H
