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

//Todo: add an error when try to get elements out of range

template <typename T>
class Matrix {
public:
    //Constructor and destructer
    Matrix(int64_t rows, int64_t cols);
    Matrix(const Matrix& other); // for copy
    Matrix(Matrix&& other) noexcept; // for move
    Matrix(T *data, int64_t rows, int64_t cols, bool isTake);
    Matrix(Matrix& parent, int64_t rowStart, int64_t colStart, int64_t rowEnd, int64_t colEnd); // for ROI
    ~Matrix();

    // for operator=
    Matrix& operator=(const Matrix& other); // for copy
    Matrix& operator=(Matrix&& other) noexcept; // for move

    // for comparsion
    // only matrices with the same size are supported!
    bool operator==(const Matrix& other) const;
    bool operator!=(const Matrix& other) const;

    // for arthimetic
    // only matrices with correct size are supported!
    Matrix operator+(const Matrix<T>& other) const;
    Matrix operator-(const Matrix<T>& other) const;
    Matrix operator*(const Matrix<T>& other) const;


    // look up for the matrix
    T& at(int64_t row, int64_t col); // look for element
    T& at(int64_t pos); // directly look for corresponding position of the array
    const T& at(int64_t row, int64_t col) const;
    const T& at(int64_t pos) const;

    const int64_t getRow() const;
    const int64_t getCol() const;

    const bool isROI() const;

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
    int64_t ROIrows_;
    int64_t ROIcols_;
    std::shared_ptr<T> data_;

    __float128 tol_; // Judging for equlity

    static const int8_t thread_num_ = 16; // thread numbers

    void init(int64_t row, int64_t col);

    static void thread_add(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const int64_t end);
    static void thread_sub(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const int64_t end);
    static void thread_mul(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const bool isRow);
};

template <typename T>
void Matrix<T>::init(int64_t row, int64_t col) {
    if constexpr (
    std::is_same<T, int8_t>::value ||
    std::is_same<T, int16_t>::value ||
    std::is_same<T, int32_t>::value ||
    std::is_same<T, int64_t>::value ||
    std::is_same<T, _Float32>::value ||
    std::is_same<T, _Float64>::value ||
    std::is_same<T, uint8_t>::value ||
    std::is_same<T, uint16_t>::value ||
    std::is_same<T, uint32_t>::value ||
    std::is_same<T, uint64_t>::value) {
    } else {
        throw std::invalid_argument("The matrix element is not supported");
    }


    if constexpr (std::is_same<T, _Float32>::value) {
        tol_ = 3 * row * col * std::numeric_limits<_Float32>::epsilon();
    } else if constexpr (std::is_same<T, _Float64>::value) {
        tol_ = 3 * row * col * std::numeric_limits<_Float64>::epsilon();
    } else {
        tol_ = 0.0f;
    }
}

template <typename T>
Matrix<T>::Matrix(int64_t rows, int64_t cols) {
    init(rows, cols);

    rows_ = rows;
    cols_ = cols;
    rows_offset_ = 0;
    cols_offset_ = 0;
    ROIrows_ = 0;
    ROIcols_ = 0;

    data_ = std::shared_ptr<T>(new(std::align_val_t(32)) T[rows_ * cols_]);
    //std::cout << "Address of first element: " << reinterpret_cast<uintptr_t>(data_.get()) << " " << reinterpret_cast<uintptr_t>(data_.get())  % 32 << std::endl;
}

template <typename T>
Matrix<T>::Matrix(const Matrix& other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    rows_offset_ = other.rows_offset_;
    cols_offset_ = other.cols_offset_;
    ROIrows_ = other.ROIrows_;
    ROIcols_ = other.ROIcols_;
    tol_ = other.tol_;

    data_ = std::shared_ptr<T>(new(std::align_val_t(32)) T[rows_ * cols_]);
    std::copy(other.data_.get(), other.data_.get() + rows_ * cols_, this->data_.get());
}

template <typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept {
    rows_ = other.rows_;
    cols_ = other.cols_;
    rows_offset_ = other.rows_offset_;
    cols_offset_ = other.cols_offset_;
    ROIrows_ = other.ROIrows_;
    ROIcols_ = other.ROIcols_;
    tol_ = other.tol_;
    data_ = std::shared_ptr<T>(other.data_);
}

template<typename T>
Matrix<T>::Matrix(Matrix& parent, int64_t rowStart, int64_t rowEnd, int64_t colStart, int64_t colEnd) {
    if (rowStart < 0 || colStart < 0 || rowEnd > parent.getRow() || colEnd > parent.getRow() || rowStart > rowEnd || colStart > colEnd) {
        throw std::invalid_argument("Invalid ROI bounds");
    }

    rows_ = parent.rows_;
    cols_ = parent.cols_;
    ROIrows_ = rowEnd - rowStart + 1;
    ROIcols_ = colEnd - colStart + 1;
    rows_offset_ = rowStart + parent.rows_offset_;
    cols_offset_ = colStart + parent.cols_offset_;
    data_ = std::shared_ptr<T>(parent.data_);
    tol_ = parent.tol_;
}

template <typename T>
Matrix<T>::Matrix(T *data, int64_t rows, int64_t cols, bool isTake) {
    init(rows, cols);

    rows_ = rows;
    cols_ = cols;
    rows_offset_ = 0;
    cols_offset_ = 0;
    ROIrows_ = 0;
    ROIcols_ = 0;



    if (isTake) {
        data_ = std::shared_ptr<T>(data, std::default_delete<T>());
    } else {
        data_ = std::shared_ptr<T>(new(std::align_val_t(32)) T[rows_ * cols_]);
        std::copy(data, data + rows_ * cols_, this->data_.get());
    }
    
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
        this->ROIrows_ = other.ROIrows_;
        this->ROIcols_ = other.ROIcols_;
        data_ = std::shared_ptr<T>(new(std::align_val_t(32)) T[rows_ * cols_]);
        std::copy(other.data_.get(), other.data_.get() + rows_ * cols_, this->data_.get());
    } else {
        std::cerr << "= is used for two identical objects." << std::endl;
    }
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        this->data_.reset();
        this->rows_ = other.rows_;
        this->cols_ = other.cols_;
        this->ROIrows_ = other.ROIrows_;
        this->ROIcols_ = other.ROIcols_;
        this->data_ = std::shared_ptr<T>(other.data_);
    } else {
        std::cerr << "= is used for two identical objects." << std::endl;
    }
    return *this;
}

template <typename T>
bool Matrix<T>::operator==(const Matrix& other) const {
    if (getRow() != other.getRow() || getCol() != other.getCol()) {
        std::cerr << "Matrices with different size are compared." << std::endl;
        return false;
    }


    __float128 tol = tol_ > other.tol_ ? tol_ : other.tol_;
    tol = 3 * getRow() * getCol() * tol;
    for (int64_t i = 0; i < getRow() * getCol(); i++) {
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
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (getRow() != other.getRow() || getCol() != other.getCol()) {
        throw std::invalid_argument("Matrices with different sizes cannot be added.");
    }

    Matrix ans = Matrix(getRow(), getCol());
    int64_t unit = getRow() * getCol() / thread_num_;

    if (getRow() * getCol() % thread_num_ != 0) {
        unit++;
    }
    std::thread* threads_ = new std::thread[thread_num_];
    
    for (int64_t i = 0; i < thread_num_; i++) {
        if (i * unit >= getRow() * getCol()) {
            break;
        }
        threads_[i] = std::thread([&ans, this, &other, i, unit]() {
            this->thread_add(ans, *this, other, i * unit, (i + 1) * unit);
        });
    }

    for (int64_t i = 0; i < thread_num_; i++) {
        if (i * unit >= getRow() * getCol()) {
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
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
    if (getRow() != other.getRow() || getCol() != other.getCol()) {
        throw std::invalid_argument("Matrices with different sizes cannot be subtracted.");
    }

    Matrix ans = Matrix(getRow(), getCol());
    int64_t unit = getRow() * getCol() / thread_num_;

    if (getRow() * getCol() % thread_num_ != 0) {
        unit++;
    }
    std::thread* threads_ = new std::thread[thread_num_];
    
    for (int64_t i = 0; i < thread_num_; i++) {
        if (i * unit >= getRow() * getCol()) {
            break;
        }
        threads_[i] = std::thread([&ans, this, &other, i, unit]() {
            this->thread_sub(ans, *this, other, i * unit, (i + 1) * unit);
        });
    }

    for (int64_t i = 0; i < thread_num_; i++) {
        if (i * unit >= getRow() * getCol()) {
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
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    if (getCol() != other.getRow()) {
        throw std::invalid_argument("Matrices with incorrect size are multiplied.");
    }

    Matrix ans = Matrix(getRow(), other.getCol());
    std::thread* threads_ = new std::thread[thread_num_];
    std::fill_n(ans.data_.get(), getRow() * other.getCol(), static_cast<T>(0));
    for (int i = 0; i < thread_num_; i++) {
        threads_[i] = std::thread([&, this, i]() {
            this->thread_mul(ans, *this, other, i, getRow() >= other.getCol());
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
    //printf("find %ld\n", (row + rows_offset_) * getCol() + col + cols_offset_);
    if (row >= getRow() || col >= getCol()) {
        throw std::invalid_argument("Matrix index out of bound");
    }
    return data_.get()[(row + rows_offset_) * cols_ + col + cols_offset_];
}

template <typename T>
const T& Matrix<T>::at(int64_t row, int64_t col) const {
    //printf("find %ld\n", (row + rows_offset_) * getCol() + col + cols_offset_);
    if (row >= getRow() || col >= getCol()) {
        throw std::invalid_argument("Matrix index out of bound");
    }
    return data_.get()[(row + rows_offset_) * cols_ + col + cols_offset_];
}

template <typename T>
T& Matrix<T>::at(int64_t pos) {
    if (pos >= getRow() * getCol()) {
        throw std::invalid_argument("Matrix index out of bound");
    }
    return data_.get()[pos + rows_offset_ * cols_ + cols_offset_];
}

template <typename T>
const T& Matrix<T>::at(int64_t pos) const {
    if (pos >= getRow() * getCol()) {
        throw std::invalid_argument("Matrix index out of bound");
    }
    //printf("finding %ld \n", pos + rows_offset_ * getCol() + cols_offset_);
    return data_.get()[pos + rows_offset_ * cols_ + cols_offset_];
}

template <typename T>
const int64_t Matrix<T>::getRow() const {
    return isROI() ? ROIrows_ : rows_;
}

template <typename T>
const int64_t Matrix<T>::getCol() const {
    return isROI() ? ROIcols_ : cols_;
}

template <typename T>
const bool Matrix<T>::isROI() const {
    return !(rows_offset_ == 0 && cols_offset_ == 0);
}

template <typename T>
void Matrix<T>::read_file(std::ifstream& file) {
    if (!file) {
        std::cerr << "Error opening file for reading." << std::endl;
        return;
    }
    file.read(reinterpret_cast<char*>(data_.get()), 32 * getRow() *getCol());
}

template <typename T>
void Matrix<T>::read_command() {
    for (int64_t i = 0; i < getRow() * getCol(); i++) {
        std::cin << at(i);
    }
}


template <typename T>
void Matrix<T>::write_file(std::ofstream& file) {
    if (!file) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(data_.get()), 32);
}

template <typename T>
void Matrix<T>::write_command() {
    for (int64_t i = 0; i < getRow(); i++) {
        for (int64_t j = 0; j < getCol(); j++) {
            std::cout << (long double)at(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void Matrix<T>::thread_add(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const int64_t end) {
    for (int64_t i = start; i < end && i < src1.getRow() * src1.getCol(); i++) {
        dest.at(i) = src1.at(i) + src2.at(i);
    }
}

template <typename T>
void Matrix<T>::thread_sub(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const int64_t end) {
    for (int64_t i = start; i < end && i < src1.getRow() * src1.getCol(); i++) {
        dest.at(i) = src1.at(i) - src2.at(i);
    }
}

template <typename T>
void Matrix<T>::thread_mul(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const bool isRow) {
    if (isRow) { // 如果行数更多，按行分割
        if (start >= src1.getRow()) { // 如果起点超出范围，直接返回
            return;
        }
            // 三重循环，计算从start_到start_ + num行的结果
        for (int64_t i = start; i < src1.getRow(); i += thread_num_) {
            for (int64_t j = 0; j < src2.getCol(); j++) {
                for (int64_t k = 0; k < src1.getCol(); k++) {
                    dest.at(i, j) += src1.at(i, k) * src2.at(k, j);
                }
            }
        }

    } else {// 列数更多，则按列分割
        if (start >= src2.getCol()) {
            return;
        }
        for (int64_t i = start; i < src2.getCol(); i += thread_num_) {
            for (int64_t j = 0; j < src1.getRow(); j++) {
                for (int64_t k = 0; k < src1.getCol(); k++) {
                    dest.at(i, j) += src1.at(i, k) * src2.at(k, j);
                }
            }
        }

    }
}

template <>
void Matrix<int8_t>::thread_add(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const int64_t end) {
    int64_t num = (std::min(end, src1.getRow() * src1.getCol()) - start) / 128;
    __m256i ymm[12];
    for (int64_t i = 0; i < num; i++) {
        ymm[0] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 128));
        ymm[1] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 128 + 32));
        ymm[2] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 128 + 64));
        ymm[3] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 128 + 96));

        ymm[4] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 128));
        ymm[5] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 128 + 32));
        ymm[6] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 128 + 64));
        ymm[7] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 128 + 96));

        ymm[8] = _mm256_add_epi8(ymm[0], ymm[4]);
        ymm[9] = _mm256_add_epi8(ymm[1], ymm[5]);
        ymm[10] = _mm256_add_epi8(ymm[2], ymm[6]);
        ymm[11] = _mm256_add_epi8(ymm[3], ymm[7]);

        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 128), ymm[8]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 128 + 32), ymm[9]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 128 + 64), ymm[10]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 128 + 96), ymm[11]);
    }

    for (int64_t i = start + num * 128; i < end && i < src1.getRow() * src1.getCol(); i++) {
        dest.at(i) = src1.at(i) + src2.at(i);
    }
}

template <>
void Matrix<int32_t>::thread_add(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const int64_t end) {
    int64_t num = (std::min(end, src1.getRow() * src1.getCol()) - start) / 32;
    __m256i ymm[12];
    for (int64_t i = 0; i < num; i++) {
        ymm[0] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 32));
        ymm[1] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 32 + 8));
        ymm[2] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 32 + 16));
        ymm[3] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 32 + 24));

        ymm[4] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 32));
        ymm[5] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 32 + 8));
        ymm[6] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 32 + 16));
        ymm[7] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 32 + 24));

        ymm[8] = _mm256_add_epi32(ymm[0], ymm[4]);
        ymm[9] = _mm256_add_epi32(ymm[1], ymm[5]);
        ymm[10] = _mm256_add_epi32(ymm[2], ymm[6]);
        ymm[11] = _mm256_add_epi32(ymm[3], ymm[7]);

        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 32), ymm[8]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 32 + 8), ymm[9]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 32 + 16), ymm[10]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 32 + 24), ymm[11]);
    }

    for (int64_t i = start + num * 32; i < end && i < src1.getRow() * src1.getCol(); i++) {
        dest.at(i) = src1.at(i) + src2.at(i);
    }
}

template <>
void Matrix<int64_t>::thread_add(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const int64_t end) {
    int64_t num = (std::min(end, src1.getRow() * src1.getCol()) - start) / 16;
    __m256i ymm[12];
    for (int64_t i = 0; i < num; i++) {
        ymm[0] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 16));
        ymm[1] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 16 + 4));
        ymm[2] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 16 + 8));
        ymm[3] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 16 + 12));

        ymm[4] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 16));
        ymm[5] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 16 + 4));
        ymm[6] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 16 + 8));
        ymm[7] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 16 + 12));

        ymm[8] = _mm256_add_epi64(ymm[0], ymm[4]);
        ymm[9] = _mm256_add_epi64(ymm[1], ymm[5]);
        ymm[10] = _mm256_add_epi64(ymm[2], ymm[6]);
        ymm[11] = _mm256_add_epi64(ymm[3], ymm[7]);

        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 16), ymm[8]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 16 + 4), ymm[9]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 16 + 8), ymm[10]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 16 + 12), ymm[11]);
    }

    for (int64_t i = start + num * 16; i < end && i < src1.getRow() * src1.getCol(); i++) {
        dest.at(i) = src1.at(i) + src2.at(i);
    }
}

template <>
void Matrix<_Float32>::thread_add(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const int64_t end) {
    int64_t num = (std::min(end, src1.getRow() * src1.getCol()) - start) / 32;
    __m256 ymm[12];
    for (int64_t i = 0; i < num; i++) {
        ymm[0] = _mm256_loadu_ps(&src1.at(start + i * 32));
        ymm[1] = _mm256_loadu_ps(&src1.at(start + i * 32 + 8));
        ymm[2] = _mm256_loadu_ps(&src1.at(start + i * 32 + 16));
        ymm[3] = _mm256_loadu_ps(&src1.at(start + i * 32 + 24));

        ymm[4] = _mm256_loadu_ps(&src2.at(start + i * 32));
        ymm[5] = _mm256_loadu_ps(&src2.at(start + i * 32 + 8));
        ymm[6] = _mm256_loadu_ps(&src2.at(start + i * 32 + 16));
        ymm[7] = _mm256_loadu_ps(&src2.at(start + i * 32 + 24));

        ymm[8] = _mm256_add_ps(ymm[0], ymm[4]);
        ymm[9] = _mm256_add_ps(ymm[1], ymm[5]);
        ymm[10] = _mm256_add_ps(ymm[2], ymm[6]);
        ymm[11] = _mm256_add_ps(ymm[3], ymm[7]);

        _mm256_storeu_ps(&dest.at(start + i * 32), ymm[8]);
        _mm256_storeu_ps(&dest.at(start + i * 32 + 8), ymm[9]);
        _mm256_storeu_ps(&dest.at(start + i * 32 + 16), ymm[10]);
        _mm256_storeu_ps(&dest.at(start + i * 32 + 24), ymm[11]);
    }

    for (int64_t i = start + num * 32; i < end && i < src1.getRow() * src1.getCol(); i++) {
        dest.at(i) = src1.at(i) + src2.at(i);
    }
}

template <>
void Matrix<_Float64>::thread_add(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const int64_t end) {
    int64_t num = (std::min(end, src1.getRow() * src1.getCol()) - start) / 16;
    __m256d ymm[12];
    //printf("start is %ld, end is %ld, num is %ld\n", start, end, num);
    for (int64_t i = 0; i < num; i++) {
        ymm[0] = _mm256_loadu_pd(&src1.at(start + i * 16));
        ymm[1] = _mm256_loadu_pd(&src1.at(start + i * 16 + 4));
        ymm[2] = _mm256_loadu_pd(&src1.at(start + i * 16 + 8));
        ymm[3] = _mm256_loadu_pd(&src1.at(start + i * 16 + 12));

        ymm[4] = _mm256_loadu_pd(&src2.at(start + i * 16));
        ymm[5] = _mm256_loadu_pd(&src2.at(start + i * 16 + 4));
        ymm[6] = _mm256_loadu_pd(&src2.at(start + i * 16 + 8));
        ymm[7] = _mm256_loadu_pd(&src2.at(start + i * 16 + 12));

        ymm[8] = _mm256_add_pd(ymm[0], ymm[4]);
        ymm[9] = _mm256_add_pd(ymm[1], ymm[5]);
        ymm[10] = _mm256_add_pd(ymm[2], ymm[6]);
        ymm[11] = _mm256_add_pd(ymm[3], ymm[7]);

        _mm256_storeu_pd(&dest.at(start + i * 16), ymm[8]);
        _mm256_storeu_pd(&dest.at(start + i * 16 + 4), ymm[9]);
        _mm256_storeu_pd(&dest.at(start + i * 16 + 8), ymm[10]);
        _mm256_storeu_pd(&dest.at(start + i * 16 + 12), ymm[11]);
    }

    for (int64_t i = start + num * 16; i < end && i < src1.getRow() * src1.getCol(); i++) {
        dest.at(i) = src1.at(i) + src2.at(i);
    }
}

template <>
void Matrix<int8_t>::thread_sub(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const int64_t end) {
    int64_t num = (std::min(end, src1.getRow() * src1.getCol()) - start) / 128;
    __m256i ymm[12];
    for (int64_t i = 0; i < num; i++) {
        ymm[0] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 128));
        ymm[1] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 128 + 32));
        ymm[2] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 128 + 64));
        ymm[3] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 128 + 96));

        ymm[4] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 128));
        ymm[5] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 128 + 32));
        ymm[6] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 128 + 64));
        ymm[7] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 128 + 96));

        ymm[8] = _mm256_sub_epi8(ymm[0], ymm[4]);
        ymm[9] = _mm256_sub_epi8(ymm[1], ymm[5]);
        ymm[10] = _mm256_sub_epi8(ymm[2], ymm[6]);
        ymm[11] = _mm256_sub_epi8(ymm[3], ymm[7]);

        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 128), ymm[8]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 128 + 32), ymm[9]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 128 + 64), ymm[10]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 128 + 96), ymm[11]);
    }

    for (int64_t i = start + num * 128; i < end && i < src1.getRow() * src1.getCol(); i++) {
        dest.at(i) = src1.at(i) - src2.at(i);
    }
}

template <>
void Matrix<int32_t>::thread_sub(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const int64_t end) {
    int64_t num = (std::min(end, src1.getRow() * src1.getCol()) - start) / 32;
    __m256i ymm[12];
    for (int64_t i = 0; i < num; i++) {
        ymm[0] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 32));
        ymm[1] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 32 + 8));
        ymm[2] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 32 + 16));
        ymm[3] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 32 + 24));

        ymm[4] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 32));
        ymm[5] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 32 + 8));
        ymm[6] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 32 + 16));
        ymm[7] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 32 + 24));

        ymm[8] = _mm256_sub_epi32(ymm[0], ymm[4]);
        ymm[9] = _mm256_sub_epi32(ymm[1], ymm[5]);
        ymm[10] = _mm256_sub_epi32(ymm[2], ymm[6]);
        ymm[11] = _mm256_sub_epi32(ymm[3], ymm[7]);

        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 32), ymm[8]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 32 + 8), ymm[9]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 32 + 16), ymm[10]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 32 + 24), ymm[11]);
    }

    for (int64_t i = start + num * 32; i < end && i < src1.getRow() * src1.getCol(); i++) {
        dest.at(i) = src1.at(i) - src2.at(i);
    }
}

template <>
void Matrix<int64_t>::thread_sub(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const int64_t end) {
    int64_t num = (std::min(end, src1.getRow() * src1.getCol()) - start) / 16;
    __m256i ymm[12];
    for (int64_t i = 0; i < num; i++) {
        ymm[0] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 16));
        ymm[1] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 16 + 4));
        ymm[2] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 16 + 8));
        ymm[3] = _mm256_loadu_si256((__m256i*)&src1.at(start + i * 16 + 12));

        ymm[4] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 16));
        ymm[5] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 16 + 4));
        ymm[6] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 16 + 8));
        ymm[7] = _mm256_loadu_si256((__m256i*)&src2.at(start + i * 16 + 12));

        ymm[8] = _mm256_sub_epi64(ymm[0], ymm[4]);
        ymm[9] = _mm256_sub_epi64(ymm[1], ymm[5]);
        ymm[10] = _mm256_sub_epi64(ymm[2], ymm[6]);
        ymm[11] = _mm256_sub_epi64(ymm[3], ymm[7]);

        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 16), ymm[8]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 16 + 4), ymm[9]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 16 + 8), ymm[10]);
        _mm256_storeu_si256((__m256i*)&dest.at(start + i * 16 + 12), ymm[11]);
    }

    for (int64_t i = start + num * 16; i < end && i < src1.getRow() * src1.getCol(); i++) {
        dest.at(i) = src1.at(i) - src2.at(i);
    }
}

template <>
void Matrix<_Float32>::thread_sub(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const int64_t end) {
    int64_t num = (std::min(end, src1.getRow() * src1.getCol()) - start) / 32;
    __m256 ymm[12];
    for (int64_t i = 0; i < num; i++) {
        ymm[0] = _mm256_loadu_ps(&src1.at(start + i * 32));
        ymm[1] = _mm256_loadu_ps(&src1.at(start + i * 32 + 8));
        ymm[2] = _mm256_loadu_ps(&src1.at(start + i * 32 + 16));
        ymm[3] = _mm256_loadu_ps(&src1.at(start + i * 32 + 24));

        ymm[4] = _mm256_loadu_ps(&src2.at(start + i * 32));
        ymm[5] = _mm256_loadu_ps(&src2.at(start + i * 32 + 8));
        ymm[6] = _mm256_loadu_ps(&src2.at(start + i * 32 + 16));
        ymm[7] = _mm256_loadu_ps(&src2.at(start + i * 32 + 24));

        ymm[8] = _mm256_sub_ps(ymm[0], ymm[4]);
        ymm[9] = _mm256_sub_ps(ymm[1], ymm[5]);
        ymm[10] = _mm256_sub_ps(ymm[2], ymm[6]);
        ymm[11] = _mm256_sub_ps(ymm[3], ymm[7]);

        _mm256_storeu_ps(&dest.at(start + i * 32), ymm[8]);
        _mm256_storeu_ps(&dest.at(start + i * 32 + 8), ymm[9]);
        _mm256_storeu_ps(&dest.at(start + i * 32 + 16), ymm[10]);
        _mm256_storeu_ps(&dest.at(start + i * 32 + 24), ymm[11]);
    }

    for (int64_t i = start + num * 32; i < end && i < src1.getRow() * src1.getCol(); i++) {
        dest.at(i) = src1.at(i) - src2.at(i);
    }
}

template <>
void Matrix<_Float64>::thread_sub(Matrix& dest, const Matrix& src1, const Matrix& src2, const int64_t start, const int64_t end) {
    int64_t num = (std::min(end, src1.getRow() * src1.getCol()) - start) / 16;
    __m256d ymm[12];
    //printf("start is %ld, end is %ld, num is %ld\n", start, end, num);
    for (int64_t i = 0; i < num; i++) {
        ymm[0] = _mm256_loadu_pd(&src1.at(start + i * 16));
        ymm[1] = _mm256_loadu_pd(&src1.at(start + i * 16 + 4));
        ymm[2] = _mm256_loadu_pd(&src1.at(start + i * 16 + 8));
        ymm[3] = _mm256_loadu_pd(&src1.at(start + i * 16 + 12));

        ymm[4] = _mm256_loadu_pd(&src2.at(start + i * 16));
        ymm[5] = _mm256_loadu_pd(&src2.at(start + i * 16 + 4));
        ymm[6] = _mm256_loadu_pd(&src2.at(start + i * 16 + 8));
        ymm[7] = _mm256_loadu_pd(&src2.at(start + i * 16 + 12));

        ymm[8] = _mm256_sub_pd(ymm[0], ymm[4]);
        ymm[9] = _mm256_sub_pd(ymm[1], ymm[5]);
        ymm[10] = _mm256_sub_pd(ymm[2], ymm[6]);
        ymm[11] = _mm256_sub_pd(ymm[3], ymm[7]);

        _mm256_storeu_pd(&dest.at(start + i * 16), ymm[8]);
        _mm256_storeu_pd(&dest.at(start + i * 16 + 4), ymm[9]);
        _mm256_storeu_pd(&dest.at(start + i * 16 + 8), ymm[10]);
        _mm256_storeu_pd(&dest.at(start + i * 16 + 12), ymm[11]);
    }

    for (int64_t i = start + num * 16; i < end && i < src1.getRow() * src1.getCol(); i++) {
        dest.at(i) = src1.at(i) - src2.at(i);
    }
}

template <>
void Matrix<int32_t>::thread_mul(Matrix<int32_t>& dest, const Matrix<int32_t>& src1, const Matrix<int32_t>& src2, const int64_t start, const bool isRow) {
    const int block_size = 8;

    if (isRow) {
        if (start >= src1.getRow()) return;

        for (int64_t i = start; i < src1.getRow(); i += thread_num_) {
            for (int64_t j = 0; j < src2.getCol() / block_size * block_size; j += block_size) {
                __m256i sum = _mm256_setzero_si256();
                for (int64_t k = 0; k < src1.getCol(); k++) {
                    __m256i a = _mm256_set1_epi32(src1.at(i, k));
                    __m256i b = _mm256_loadu_si256((__m256i*)&src2.at(k, j));
                    __m256i prod = _mm256_mullo_epi32(a, b);
                    sum = _mm256_add_epi32(sum, prod);
                }
                _mm256_storeu_si256((__m256i*)&dest.at(i, j), sum);
            }
            // Handle remainder of columns
            for (int64_t j = src2.getCol() / block_size * block_size; j < src2.getCol(); ++j) {
                int32_t sum = 0;
                for (int64_t k = 0; k < src1.getCol(); k++) {
                    sum += src1.at(i, k) * src2.at(k, j);
                }
                dest.at(i, j) = sum;
            }
        }
    } else {
        if (start >= src2.getCol()) return;

        for (int64_t j = start; j < src2.getCol(); j += thread_num_) {
            for (int64_t i = 0; i < src1.getRow(); i++) {
                __m256i sum = _mm256_setzero_si256();
                for (int64_t k = 0; k < src1.getCol(); k++) {
                    __m256i a = _mm256_set1_epi32(src1.at(i, k));
                    __m256i b = _mm256_loadu_si256((__m256i*)&src2.at(k, j));
                    __m256i prod = _mm256_mullo_epi32(a, b);
                    sum = _mm256_add_epi32(sum, prod);
                }
                // Check if we need to store less than 8 elements
                if (j + block_size <= src2.getCol()) {
                    _mm256_storeu_si256((__m256i*)&dest.at(i, j), sum);
                } else {
                    // Store the remaining elements one by one
                    for (int rem = 0; rem < src2.getCol() % block_size; ++rem) {
                        dest.at(i, j + rem) = sum[rem]; // pseudo-code for direct element access in _mm256
                    }
                }
            }
        }
    }
}



template <>
void Matrix<_Float32>::thread_mul(Matrix<_Float32>& dest, const Matrix<_Float32>& src1, const Matrix<_Float32>& src2, const int64_t start, const bool isRow) {
    const int block_size = 8;

    if (isRow) {
        if (start >= src1.getRow()) return;

        for (int64_t i = start; i < src1.getRow(); i += thread_num_) {
            for (int64_t j = 0; j < src2.getCol() / block_size * block_size; j += block_size) {
                __m256 sum = _mm256_setzero_ps();
                for (int64_t k = 0; k < src1.getCol(); k++) {
                    __m256 a = _mm256_broadcast_ss(&src1.at(i, k));
                    __m256 b = _mm256_loadu_ps(&src2.at(k, j));
                    sum = _mm256_fmadd_ps(a, b, sum);
                }
                _mm256_storeu_ps(&dest.at(i, j), sum);
            }
            // Handle remainder of columns
            for (int64_t j = src2.getCol() / block_size * block_size; j < src2.getCol(); ++j) {
                _Float32 sum = 0.0;
                for (int64_t k = 0; k < src1.getCol(); k++) {
                    sum += src1.at(i, k) * src2.at(k, j);
                }
                dest.at(i, j) = sum;
            }
        }
    } else {
        if (start >= src2.getCol()) return;

        for (int64_t j = start; j < src2.getCol(); j += thread_num_) {
            for (int64_t i = 0; i < src1.getRow(); i++) {
                __m256 sum = _mm256_setzero_ps();
                for (int64_t k = 0; k < src1.getCol(); k++) {
                    __m256 a = _mm256_broadcast_ss(&src1.at(i, k));
                    __m256 b = _mm256_loadu_ps(&src2.at(k, j));
                    sum = _mm256_fmadd_ps(a, b, sum);
                }
                // Check if we need to store less than 8 elements
                if (j + block_size <= src2.getCol()) {
                    _mm256_storeu_ps(&dest.at(i, j), sum);
                } else {
                    // Store the remaining elements one by one
                    for (int rem = 0; rem < src2.getCol() % block_size; ++rem) {
                        dest.at(i, j + rem) = sum[rem]; // pseudo-code for direct element access in _mm256
                    }
                }
            }
        }
    }
}

template <>
void Matrix<_Float64>::thread_mul(Matrix<_Float64>& dest, const Matrix<_Float64>& src1, const Matrix<_Float64>& src2, const int64_t start, const bool isRow) {
    const int block_size = 4;

    if (isRow) {
        if (start >= src1.getRow()) return;

        for (int64_t i = start; i < src1.getRow(); i += thread_num_) {
            for (int64_t j = 0; j < src2.getCol() / block_size * block_size; j += block_size) {
                __m256d sum = _mm256_setzero_pd();
                for (int64_t k = 0; k < src1.getCol(); k++) {
                    __m256d a = _mm256_set1_pd(src1.at(i, k));
                    __m256d b = _mm256_loadu_pd(&src2.at(k, j));
                    __m256d prod = _mm256_mul_pd(a, b);
                    sum = _mm256_add_pd(sum, prod);
                }
                _mm256_storeu_pd(&dest.at(i, j), sum);
            }
            // 处理剩余列
            for (int64_t j = src2.getCol() / block_size * block_size; j < src2.getCol(); ++j) {
                _Float64 sum = 0.0;
                for (int64_t k = 0; k < src1.getCol(); k++) {
                    sum += src1.at(i, k) * src2.at(k, j);
                }
                dest.at(i, j) = sum;
            }
        }
    } else {
        if (start >= src2.getCol()) return;

        for (int64_t j = start; j < src2.getCol(); j += thread_num_) {
            for (int64_t i = 0; i < src1.getRow(); i++) {
                __m256d sum = _mm256_setzero_pd();
                for (int64_t k = 0; k < src1.getCol(); k++) {
                    __m256d a = _mm256_set1_pd(src1.at(i, k));
                    __m256d b = _mm256_loadu_pd(&src2.at(k, j));
                    __m256d prod = _mm256_mul_pd(a, b);
                    sum = _mm256_add_pd(sum, prod);
                }
                _mm256_storeu_pd(&dest.at(i, j), sum);
            }
        }
    }
}

#endif // MATRIX_H
