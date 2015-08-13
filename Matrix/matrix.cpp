
#ifndef _Matrix_DLL
#define _Matrix_DLL // __declspec(dllexport)
#endif

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include <memory>
#include <initializer_list>
#include <iostream>

#include "../refs.h"
#include "../math/rect.h"
#include "matrix.h"

#include <omp.h>

#ifndef NULL
#define NULL 0
#endif

#pragma warning(disable : 4996) // see the pop at bottom of file

/* --------------------------------------------------------------- */

#ifdef MAT_ENABLE_PARALLEL
#	ifndef _MAT_ENABLE_PARALLEL
#		define _MAT_ENABLE_PARALLEL
#ifdef _WIN32
#		pragma message("OPENMP - disabled")
#else
#		warning "OPENMP - enabled"
#endif
#	endif
#else
#ifdef _WIN32
#pragma message("OPENMP - disabled")
#else
#warning "OPENMP - disabled"
#endif
#endif


/* --------------------------------------------------------------- */

namespace ml {

template <typename T>
Matrix_T<T>* matrix_create(int height, int width) {
    Matrix_T<T>* mat = NULL;
    if ( height <= 0 || width <= 0 )
        return mat;
    mat = new Matrix_T<T>();
    if (!matrix_resize(mat, height, width, false)) {
        delete mat;
    mat = NULL;
  }
    return mat;
}

template <typename T>
Matrix_T<T>* matrix_create(int height, int width, T value) {
    Matrix_T<T>* mat = matrix_create<T>(height, width);
    if (!mat) {
        return mat;
    }
    if (!matrix_init<T>(mat, value)) {
        delete mat;
        mat = NULL;
    }
  return mat;
}

template <typename T>
bool matrix_resize(Matrix_T<T>* matrix, int height, int width,
    bool bForceInitZero/*=false*/)
{
  if (!matrix) {
    return false;
  }
  matrix->row_count = height;
  matrix->col_count = width;

    // validation
    assert(matrix);
    if (matrix->row_count <= 0 || matrix->col_count <= 0) {
        return false;
  }

  /* if values already exists then clear it.. we are resizing..*/
  if (matrix->values) {
    if (matrix->row_count > 0 && matrix->values[0]) {
        const int nRows = matrix->row_count;
        #pragma omp parallel for
      for (int i = 0; i < nRows; ++i) {
        delete[] matrix->values[i];
      }
    }
    delete[] matrix->values;
  }


  /* Re-create the mat's values */
  matrix->values = matrix_alloc_rows<T>(matrix->row_count);
    #pragma omp parallel for
  for (int i = 0; i < matrix->row_count; ++i) {
    matrix->values[i] = matrix_alloc_row<T>(matrix->col_count);
  }

    if ( bForceInitZero ) {
        return matrix_init<T>(matrix, 0); //returns true if successful..
    }
    return true;
}

template <typename T>
T** matrix_alloc_rows(int nRows) {
    return new T*[nRows];
}

template <typename T>
T* matrix_alloc_row(int nCols) {
    return new T[nCols];
}

/* Checks if matrix is allocated..
    if so then it sets all of the elements
    data equal to 'value'
    returns true if success
    returns false if matrix was empty
        or no rows
        or no columns   */
template <typename T>
bool matrix_init(Matrix_T<T>* matrix, T value) {
    if (!matrix || matrix->row_count <= 0) {
        return false;
    }
        assert(matrix);
        // #ifdef _MAT_ENABLE_PARALLEL
        #pragma omp parallel for
    for (int i = 0; i < matrix->row_count; ++i)
        for (int j = 0; j < matrix->col_count; ++j)
            matrix->values[i][j] = value;
    return true;
}

/* returns the total number of elements */
template <typename T>
int matrix_count_elements(Matrix_T<T>* matrix){
    return matrix->row_count * matrix->col_count;
}

/* get value at i,j.. could just matrix->value[i][j] */
template <typename T>
inline T matrix_get_value(Matrix_T<T>* matrix, int i, int j){
    return matrix->values[i][j];
}

template <typename T>
inline void matrix_set_value(Matrix_T<T>* matrix, int i, int j, T value) {
    matrix->values[i][j] = value;
}


template <typename T>
_Matrix_DLL T* matrix_get_row(Matrix_T<T>* matrix, int i) {
    const size_t nCountCols = matrix->col_count;
    T* row = new T[nCountCols];
    memcpy(row, matrix->values[i], nCountCols * sizeof(T));
    assert(row);
    return row;
}
template <typename T>
_Matrix_DLL T* matrix_get_col(Matrix_T<T>* matrix, int j) {
    const size_t nCountRows = matrix->row_count;
    T* col = new T[nCountRows];
    #pragma omp parallel for
    for (int i = 0; i < nCountRows; i++)
        col[i] = matrix_get_value(matrix, i, j);
    return col;
}

template <typename T>
_Matrix_DLL void matrix_set_row(Matrix_T<T>* matrix, int i, T* row) {
    assert(row && matrix);
    const size_t nCountCols = matrix->col_count;
    if (nCountCols <= 0)
        return;
    if (matrix->values[i] == NULL)
        matrix->values[i] = matrix_alloc_row<T>(nCountCols);
    memcpy(matrix->values[i], row, nCountCols * sizeof(T));
}
template <typename T>
_Matrix_DLL void matrix_set_col(Matrix_T<T>* matrix, int j, T* col) {
    assert(col && matrix);
    const size_t nCountRows = matrix->row_count;
    #pragma omp parallel for
    for (int i = 0; i < nCountRows; i++)
        matrix_set_value(matrix, i, j, col[i]);
}

template <typename T>
_Matrix_DLL void matrix_push_row(Matrix_T<T>* matrix, T* row) {
    const size_t nNewRows = matrix->row_count + 1; // add a row
    T** vals = matrix->values;

    /* Create new mat and copy pointers to rows */
    matrix->values = matrix_alloc_rows<T>(nNewRows);
    for (size_t i = 0; i < nNewRows-1; i++) {
        matrix->values[i] = vals[i];
        vals[i] = NULL;
    }

    // Free the old table
    delete[] vals;
    vals = NULL;

    /* Add the new row */
    matrix->values[nNewRows] = matrix_alloc_row<T>(matrix->col_count);
    memcpy(matrix->values[nNewRows], row, matrix->col_count * sizeof(T));
    matrix->row_count = nNewRows;
}

template <typename T>
_Matrix_DLL void matrix_push_col(Matrix_T<T>* matrix, T* col) {
    if (!matrix || !matrix->values || !matrix->values[0]) return;
    const size_t nNewRows = matrix->row_count;
    const size_t nNewCols = matrix->col_count + 1;

    // allocate the new values arrays and set the col value at each row while we're at it
    T** vals = matrix_alloc_rows<T>(nNewRows);
    for (int i = 0; i < nNewRows; ++i) {
        assert(matrix->values[i] != NULL);
        vals[i] = matrix_alloc_row<T>(nNewCols);
        memcpy(vals[i], matrix->values[i], matrix->col_count * sizeof(T));
        vals[i][nNewCols - 1] = col[i];
    }

    // the old matrix values do not apply, exterminaaaaate
    for (int i = 0; i < matrix->row_count; ++i)
        delete[] matrix->values[i];
    delete[] matrix->values;
    matrix->values = vals;
    vals = NULL;

    matrix->row_count = nNewRows;
    matrix->col_count = nNewCols;
}

template <typename T>
T* vector_copy(T* vec, size_t count) {
    T* dst = new T[count];
    memcpy(dst, vec, count * sizeof(T));
    return dst;
}

// range is inclusive
template <typename T>
T* vector_copy_range(T* vec, size_t a, size_t b) {
    const size_t count = b - a + 1;
    T* dst = new T[count];
    memcpy(dst, vec + a, count * sizeof(T));
    return dst;
}

// mat needs to be allocated and of the size size..
// use resize on dst if not
template <typename T>
void matrix_copy(Matrix_T<T>* src, Matrix_T<T>* dst)
{
    assert(dst->col_count >= src->col_count
            && dst->row_count >= src->row_count);
    // #ifdef _MAT_ENABLE_PARALLEL
    #pragma omp parallel for
    for (int i = 0; i < src->row_count; ++i)
        memcpy(dst->values[i], src->values[i], src->col_count * sizeof(T));
}

template <typename T>
Matrix_T<T>* matrix_copy(Matrix_T<T>* mat) {
    Matrix_T<T>* newMat = NULL;
    int height = mat->row_count;
    int width = mat->col_count;
    if (height <= 0 || width <= 0)
        return newMat;

    // create a new matrix and resize to the copied mat
    newMat = matrix_create<T>(height, width);
    if (!newMat) {
        return newMat;
    }

    // copy the values into the new mat
    matrix_copy(mat, newMat);
    return newMat;
}

/* with respect to the bottom left closest to the origin than top right */
template <typename T>
Matrix_T<T>* matrix_copy_roi(Matrix_T<T>* mat, Rect<int> roi) {
    Matrix_T<T>* newMat = new Matrix_T<T>();
    const int nRows = roi.GetHeight();
    const int nCols = roi.GetWidth();
    const int a = roi.GetBottom(); // this seems backwards but its normal..
    const int l = roi.GetLeft();
    for (int i = 0; i < nRows; ++i)
        memcpy(newMat->values[i], mat->values[a+i] + l, nCols * sizeof(T));
    newMat->row_count = nRows;
    newMat->col_count = nCols;
    return newMat;
}

template <typename T>
bool matrix_dealloc_values(Matrix_T<T>* matrix) {
    if (!matrix || !matrix->values || matrix->row_count <= 0) {
        return false;
    }
        #pragma omp parallel for
    for (int i = 0; i < matrix->row_count; ++i) {
        delete[] matrix->values[i]; // deletes a row vector
    }
    return true;
}

/* clears the matrix and frees the pointers  */
template <typename T>
bool matrix_delete(Matrix_T<T>* matrix){
    if (!matrix_dealloc_values<T>(matrix))
        return false;
    delete[] matrix->values; // free pointer to pointers
        delete matrix;
    return true;
}

template <typename T>
bool matrix_destroy(Matrix_T<T>* matrix) {
    return matrix_delete<T>(matrix);
}

template <typename T>
void matrix_print(Matrix_T<T>* matrix) {
    char comma[] = "  ";
    char brace_left = '[';
    char brace_right = ']';

    printf("%c \n", brace_left);
    for (int i = 0; i < matrix->row_count; ++i) {
        printf(" %c ", brace_left);
        strncpy(comma, ", ", 2);
        /* print the results */
        for (int j = 0; j < matrix->col_count; ++j) {
            if (j == matrix->col_count-1) {
                strncpy(comma, "  ", 2);
            }
            matrix_print_value<T>(matrix->values[i][j], comma);
        }

        strncpy(comma, ", ", 2);
        /* determine conditional string */
        if (i == matrix->row_count-1) {
            strncpy(comma, "  ", 2);
        }

        printf(" %c%s", brace_right, comma);
        printf("\n");
    }
    printf("%c\n\n", brace_right);
}

template <typename T>
void matrix_print_value(T& v, char* comma) {
    printf("%0.2f%s ", (double)v, comma);
}

template <>
void matrix_print_value<int>(int& v, char* comma) {
    printf("%0.2d%s ", v, comma);
}
template <>
void matrix_print_value<unsigned int>(unsigned int& v, char* comma) {
    printf("%0.2u%s ", v, comma);
}
template <>
void matrix_print_value<long>(long& v, char* comma) {
    printf("%0.2l%s ", v, comma);
}

template <typename T>
bool matrix_isvalid(Matrix_T<T>* matrix){
    if (!matrix || !matrix->values
            || matrix->row_count <= 0
            || !matrix->values[0]) {
        return false;
    }
    return true;
}


/* populates a matrix with random values if it is a valid matrix */
template <typename T>
bool matrix_populate_random(Matrix_T<T>* matrix, int m, int n){
    if ( !matrix_isvalid<T>(matrix)) {
        return false;
    }
        assert(matrix);
        #pragma omp parallel for
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++) {
            srand((uint)time(NULL));
            int multiplier_val = 20;
            matrix->values[i][j] = rand() %
                (4* (int)sqrt(1.*n*multiplier_val)) - multiplier_val/2;
        }
    }
    return true;
}

/* simply generates a random new matrix */
template <typename T>
Matrix_T<T>* matrix_generate_random(int m, int n){
    Matrix_T<T>* C = matrix_create<T>(m, n);
    matrix_populate_random<T>(C, m, n);
    return C;
}



/* --------------------------------------------------------------- */

/*
    Main Multiply Methods :
        Forms: ijk, ikj, kij
                                */

/* M1 * M2 --- ijk form */
template <typename T>
Matrix_T<T>* matrix_mult_ijk(Matrix_T<T>* M1, Matrix_T<T>* M2) {
    int vector_size = M1->col_count;

    /*  if vectors size, or matrix width is
        not appropriate. the number of elements
        being multiplied must be equal      */
    if (vector_size != M2->row_count) {
        return NULL;
    }

    int new_row_count = M1->row_count;
    int new_col_count = M2->col_count;

    Matrix_T<T>* M3 = matrix_create<T>(new_row_count, new_col_count, 0);
        assert(M3);

        //#pragma omp parallel for
    for (int i = 0; i < M1->row_count; ++i) {
        for (int j = 0; j < M2->col_count; ++j) {
            for (int k = 0; k < vector_size; ++k) {
                M3->values[i][j] += M1->values[i][k] * M2->values[k][j];
            }
        }
    }

    return M3;
}

template <typename T>
Matrix_T<T>*  matrix_mult_ikj(Matrix_T<T>* M1, Matrix_T<T>* M2){
    assert(M1->values && M2->values);
    int vector_size = M1->col_count;

    /*  if vectors size, or matrix width is
        not appropriate. the number of elements
        being multiplied must be equal      */
    if (vector_size != M2->row_count) {
        return NULL;
    }

    int new_row_count = M1->row_count;
    int new_col_count = M2->col_count;

    Matrix_T<T>* M3 = matrix_create<T>(new_row_count, new_col_count);

    for (int k = 0; k < vector_size; ++k) {
        for (int i = 0; i < M1->row_count; ++i) {
            for (int j = 0; j < M2->col_count; ++j) {
                M3->values[i][j] += M1->values[i][k] * M2->values[k][j];
            }
        }
    }

    return M3;
}

template <typename T>
Matrix_T<T>* matrix_mult_kij(Matrix_T<T>* M1, Matrix_T<T>* M2){
    int vector_size = M1->col_count;

    /*  if vectors size, or matrix width is
        not appropriate. the number of elements
        being multiplied must be equal      */
    if (vector_size != M2->row_count) {
        return NULL;
    }

    int new_row_count = M1->row_count;
    int new_col_count = M2->col_count;

    Matrix_T<T>* M3 = matrix_create<T>(new_row_count, new_col_count);
        #pragma omp parallel for
    for (int i = 0; i < M1->row_count; ++i) {
        for (int k = 0; k < vector_size; ++k) {
            for (int j = 0; j < M2->col_count; ++j) {
                M3->values[i][j] += M1->values[i][k] * M2->values[k][j];
            }
        }
    }
    return M3;
}

template <typename T>
bool matrix_check_values_equal(Matrix_T<T>* mat, T value) {
    if ( !matrix_isvalid<T>(mat) ) return false;
    bool bMatrixValuesEqualToValue = true;
    const T expected_value = 1;
    #pragma omp parallel for
    for (int i = 0; (i < mat->row_count); ++i) {
        for (int j = 0; (j < mat->col_count) && bMatrixValuesEqualToValue ; ++j) {
            if ( mat->values[i][j] != expected_value ) {
                bMatrixValuesEqualToValue = false;
                break;
            }
        }
    }
    return bMatrixValuesEqualToValue;
}

template <typename T>
size_t matrix_count_values_equal(Matrix_T<T>* mat, T value) {
    if ( !matrix_isvalid<T>(mat) ) return -1;
    size_t num_values_equal = 0;
    const T expected_value = 1;
    #pragma omp parallel for
    for (int i = 0; (i < mat->row_count); ++i) {
        for (int j = 0; (j < mat->col_count); ++j) {
            if ( mat->values[i][j] == value ) {
                num_values_equal += 1;
            }
        }
    }
    return num_values_equal;
}

/* Template class forward declarations are not necessary here
   since the Matrix class is already forward declared and through
     that the Matrix_T structs are also implicitly forward declared.  */

// unless we do things differently.. in which case.. these are necessary



}



///* --------------------------------------------------------------- */


namespace ml {
    template <typename T>
    Mat<T>::Mat() {
        mMat = 0;
    }

    template <typename T>
    Mat<T>::Mat(const Mat& mat) {
        mMat = mat.mMat;
    }

    template <typename T>
    Mat<T>::Mat(MatrixPtr pMat) {
        mMat = pMat;
    }

    template <typename T>
    Mat<T>::Mat(int height, int width, int val) {
        mMat = MatrixPtr(matrix_create<T>(height, width, val));
    }

    template <typename T>
    Mat<T>::Mat(const Size& size, int val) {
        mMat = MatrixPtr(matrix_create<T>(size.cy, size.cx, val));
    }

    template <typename T>
    Mat<T>::Mat(L l) {
        /* Extract size information from ll */
        size_t nCols = l.size();
        if (nCols <= 0) {
            mMat = 0;
            return;
        }
        mMat = MatrixPtr(matrix_create<T>(1, nCols));
        int j = 0;
        for (auto val : l) {
            mMat->values[0][j] = val;
            ++j;
        }
    }

    template <typename T>
    Mat<T>::Mat(LL ll) {
        /* Extract size information from ll */
        size_t nRows = ll.size();
        if (nRows <= 0) {
            mMat = 0;
            return;
        }
        size_t nCols = (*(ll.begin())).size();
        if (nCols <= 0) {
            mMat = 0;
            return;
        }

        /* Create mat from size info */
        mMat = MatrixPtr(matrix_create<T>(nRows, nCols));

        /* Init each value one by one from ll */
        int i = 0;
        for (auto l : ll) {
            int j = 0;
            for (auto val : l) {
                mMat->values[i][j] = val;
                ++j;
            }
            ++i;
        }
    }

    template <typename T>
    Mat<T>::~Mat() { }

    template <typename T>
    typename Mat<T>::Row& Mat<T>::operator[](int idx) {
        //typename Mat<T>::Row row(mMat->values[idx], mMat->values[idx] + mMat->col_count);
        //return row;
        return mMat->values[idx];
    }

    template <typename T>
    bool Mat<T>::IsGood() const {
        return (mMat != 0 && mMat->row_count > 0 && mMat->col_count > 0 && mMat->values);
    }

    template <typename T>
    Mat<T> Mat<T>::Mult(const Mat<T>& mat) const {
        return Mat<T>(matrix_mult_ijk<T>(this->mMat.get(), mat.mMat.get()));
    }

    template <typename T>
    Mat<T>& Mat<T>::Transpose() {
        if (!IsGood()) return *this;
        // naive transpose. TODO: transpose inline
        const int nNewRows = mMat->col_count;
        const int nNewCols = mMat->row_count;
        MatrixPtr pNewMat(matrix_create<T>(nNewRows, nNewCols));
        for (size_t j = 0; j < nNewRows; ++j) {
            T* col = matrix_get_col<T>(mMat.get(), j);
            matrix_set_row<T>(pNewMat.get(), j, col);
            delete col;
        }
        mMat = pNewMat;
        return *this;
    }


    template <typename T>
    Mat<T> Mat<T>::Copy() const {
        Mat<T> mat;
        mat.mMat = MatrixPtr(matrix_copy<T>(mMat.get()));
        return mat;
    }

    template <typename T>
    Mat<T>& Mat<T>::Print() {
        matrix_print<T>(this->mMat.get());
        return *this;
    }

    template <typename T>
    Mat<T> Mat<T>::CopyROI(Rect<int> roi) const {
        Mat<T> copy;
        copy.mMat = MatrixPtr(matrix_copy_roi<T>(mMat.get(), roi));
        return copy;
    }

    template <typename T>
    Mat<T>::Mat(Matrix_T<T>* mat) {
        mMat = MatrixPtr(mat);
    }

    // return row must be deleted, unless using vector
    template <typename T>
    typename Mat<T>::Row Mat<T>::row(int i) const {
        return matrix_get_row<T>(mMat.get(), i);
        //T* r = matrix_get_row<T>(mMat.get(), i);
        //typename Mat<T>::Row row(r, r + mMat->col_count);
        //delete[] r;
        //return row;
    }

    // return col must be deleted, unless using vector
    template <typename T>
    typename Mat<T>::Col Mat<T>::col(int j) const {
        return matrix_get_col<T>(mMat.get(), j);
        //T* c = matrix_get_col<T>(mMat.get(), j);
        //typename Mat<T>::Row col(c, c + mMat->row_count);
        //delete[] c;
        //return col;
    }

    // param row must be deleted, unless using vector
    template <typename T>
    void Mat<T>::row(int i, typename Mat<T>::Row row){
        //matrix_set_row<T>(mMat.get(), i, row.data());
        matrix_set_row<T>(mMat.get(), i, row);
    }

    // param col must be deleted, unless using vector
    template <typename T>
    void Mat<T>::col(int j, typename Mat<T>::Col col){
        //matrix_set_col<T>(mMat.get(), j, col.data());
        matrix_set_col<T>(mMat.get(), j, col);
    }

    template <typename T>
    void Mat<T>::pushCol(typename Mat<T>::Col col) {
        matrix_push_col<T>(mMat.get(), col);
    }

    template <typename T>
    void Mat<T>::pushRow(typename Mat<T>::Row row) {
        matrix_push_row<T>(mMat.get(), row);
    }



    //  /* - Forward Declarations - */
    template class Mat<int>;
    template class Mat<float>;
    template class Mat<double>;
    template class Mat<long long>;
    template class Mat<long double>;
    template class Mat<long unsigned int>;
}



///* ------------------------------------------------ */
///*
//    Mat<T> extensions.
//
//    Det - determinant.
//    findM - finds the M value for Det..
//*/
//


namespace ml {
        template <typename T>
        T det(const Mat<T>& A) {
            T determinant_SUBSUM = 0;
            int rowP = 0;
            if (A.size().cx == 2 && A.size().cy == 2)
                return (A.getAt(0,0) * A.getAt(1,1) - A.getAt(1,0) * A.getAt(0,1));
            for (int x_iter=0; x_iter < A.size().cx; x_iter++) {
                T r = A.getAt(rowP, x_iter) * (pow(-1, (rowP + x_iter)))
                        * det<T>( findM<T>(A, rowP, x_iter, A.size()) );
                determinant_SUBSUM += r;
            }
            return determinant_SUBSUM;
        }

        template int det(const ml::Mat<int>& A);
        template float det(const ml::Mat<float>& A);
        template double det(const ml::Mat<double>& A);
        template long long det(const ml::Mat<long long>& A);
        template long double det(const ml::Mat<long double>& A);
        template long unsigned int det(const ml::Mat<long unsigned int>& A);

        // SIZE is the size of the Matrix_T<T> (SIZE * SIZE)
        template <typename T>
        Mat<T> findM(const Mat<T>& A, int i, int j, ml::Size size) {
            Mat<T> B(size.cy-1, size.cx-1, 0);
            int new_row_iter = 0;
            int new_col_iter = 0;
            for (int row_iter = 0; row_iter < size.cy; row_iter++) {
                if (row_iter == i) {
                    continue;
                } else {
                    for(int col_iter = 0; col_iter < size.cx; col_iter++){
                        if (col_iter == j) {
                            continue;
                        } else {
                            B.setAt(new_row_iter, new_col_iter, (A.getAt(row_iter,col_iter)) );
                            new_col_iter++;
                        }
                    } // end col for
                    new_col_iter = 0;
                    new_row_iter++;
                }
            } // end row for
            return B;
        };

        template ml::Mat<int> findM(const ml::Mat<int>& A, int i, int j, ml::Size size);
        template ml::Mat<float> findM(const ml::Mat<float>& A, int i, int j, ml::Size size);
        template ml::Mat<double> findM(const ml::Mat<double>& A, int i, int j, ml::Size size);
        template ml::Mat<long long> findM(const ml::Mat<long long>& A, int i, int j, ml::Size size);
        template ml::Mat<long double> findM(const ml::Mat<long double>& A, int i, int j, ml::Size size);
        template ml::Mat<long unsigned int> findM(const ml::Mat<long unsigned int>& A, int i, int j, ml::Size size);
}
