#pragma once

#ifndef _Matrix_DLL
#define _Matrix_DLL // __declspec(dllimport)
#endif

#ifndef uint
#define uint unsigned int
#endif

#define MAT_ENABLE_PARALLEL

#include <math.h>
#include <random>
#include <functional>

/// / / / / / /// / / / / / /// / / / / / /// / / / / /

namespace ml {
    template <typename T>
    struct Matrix_T {
        Matrix_T() {
            values = NULL;
            row_count = 0;
            col_count = 0;
            z_count = 0;
        }
        virtual ~Matrix_T() {
            if (values) {
                for (int i = 0; i < row_count; i++) {
                    if (values[i]) {
                        delete[] values[i];
                        values[i] = NULL;
                    }
                }
                delete[] values;
                values = NULL;
            }
        }
        T** values;
        int row_count;
        int col_count;
        int z_count;
    };


    template <typename T>
    _Matrix_DLL Matrix_T<T>*    matrix_create(int height, int width);								// create a new matrix with width and height

    template <typename T>
    _Matrix_DLL Matrix_T<T>*    matrix_create(int height, int width, T value);				// create a new matrix with width and height

    /* Alloc helpers */
    template <typename T>
    _Matrix_DLL T**             matrix_alloc_rows(int nRows);

    template <typename T>
    _Matrix_DLL T*              matrix_alloc_row(int nCols);


    template <typename T>
    _Matrix_DLL bool        	matrix_resize(Matrix_T<T>* matrix, int height, int width, bool bForceInitZero=false);   // resize a matrix to the matrix's row and col count

    template <typename T>
    _Matrix_DLL bool        	matrix_init(Matrix_T<T>* matrix, T value);						// initialize the elements in the matrix to the value

    template <typename T>
    _Matrix_DLL int         	matrix_count_elements(Matrix_T<T>* matrix);								// returns the total number of elements in the Matrix_T<T>

    template <typename T>
    inline _Matrix_DLL T		matrix_get_value(Matrix_T<T>* matrix, int i, int j);						// returns the value in the matrix at the i and j

    template <typename T>
    inline _Matrix_DLL void		matrix_set_value(Matrix_T<T>* matrix, int i, int j, T value);

    template <typename T>
    _Matrix_DLL T*				matrix_get_row(Matrix_T<T>* matrix, int i);
    template <typename T>
    _Matrix_DLL T*				matrix_get_col(Matrix_T<T>* matrix, int j);

    template <typename T>
    _Matrix_DLL void			matrix_set_row(Matrix_T<T>* matrix, int i, T* row);
    template <typename T>
    _Matrix_DLL void			matrix_set_col(Matrix_T<T>* matrix, int j, T* col);

    template <typename T>
    _Matrix_DLL void			matrix_push_row(Matrix_T<T>* matrix, T* row);
    template <typename T>
    _Matrix_DLL void			matrix_push_col(Matrix_T<T>* matrix, T* col);

    template <typename T>
    _Matrix_DLL T* 				vector_copy(T* vec, size_t count);

    template <typename T>
    _Matrix_DLL void			matrix_copy(Matrix_T<T>* src, Matrix_T<T>* dst);

    template <typename T>
    _Matrix_DLL Matrix_T<T>*    matrix_copy(Matrix_T<T>* mat);											// copies the mat

                                                                                                    //  (or just matrix->values[i][j])
    template <typename T>
    _Matrix_DLL bool        	matrix_isvalid(Matrix_T<T>* matrix);										// determines if matrix is not null and value pointers are not null

    template <typename T>
    _Matrix_DLL bool        	matrix_dealloc_values(Matrix_T<T>*);

    template <typename T>
    _Matrix_DLL bool        	matrix_delete(Matrix_T<T>*);												// clears the memory used by the Matrix_T<T>

    template <typename T>
    _Matrix_DLL bool			matrix_destroy(Matrix_T<T>*);

    template <typename T>
    _Matrix_DLL void        	matrix_print(Matrix_T<T>* matrix);

    template <typename T>
    _Matrix_DLL void 			matrix_print_value(T& v, char* comma);

    template <typename T>
    _Matrix_DLL bool        	matrix_populate_random(Matrix_T<T>* matrix, int m, int n);				// populates an existing matrix

    template <typename T>
    _Matrix_DLL Matrix_T<T>*    matrix_generate_random(int m, int n);								// generates a new matrix and populates it, randomly

    /* Matrix_T<T> Multiply Functions */
    template <typename T>
    _Matrix_DLL Matrix_T<T>*    matrix_mult_ijk(Matrix_T<T>* M1, Matrix_T<T>* M2);

    template <typename T>
    _Matrix_DLL Matrix_T<T>*    matrix_mult_ikj(Matrix_T<T>* M1, Matrix_T<T>* M2);

    template <typename T>
    _Matrix_DLL Matrix_T<T>*    matrix_mult_kij(Matrix_T<T>* M1, Matrix_T<T>* M2);

    template <typename T>
    _Matrix_DLL bool		    matrix_check_values_equal(Matrix_T<T>* mat, T value);

    template <typename T>
    _Matrix_DLL size_t 			matrix_count_values_equal(Matrix_T<T>* mat, T value);
}

namespace ml {

    class Size {
    public:
        Size() { cx = 0; cy = 0; }
        Size(int _cx, int _cy) {
            cx = _cx;
            cy = _cy;
        }
        bool operator==(const Size& size) {
            return ((size.cx == this->cx) && (size.cy == this->cy));
        }
        bool operator!=(const Size& size) {
            return ((size.cx != this->cx) || (size.cy != this->cy));
        }
        int cx;
        int cy;
    };
}

namespace ml {
    template <typename T>
    class Mat {
    public:
        typedef std::shared_ptr<Matrix_T<T>> MatrixPtr;
        typedef T* Row;
        typedef T* Col;
        //typedef ml::refArr<T> RowRef;
        //typedef ml::refArr<T> ColRef;
        //typedef std::vector<T> Row;
        //typedef std::vector<T> Col;
        typedef typename std::initializer_list<T> L;
        typedef typename std::initializer_list<std::initializer_list<T>> LL;

    public:
        Mat();
        Mat(const Mat& mat);
        Mat(std::vector<Row> rows, const int row_size);
        Mat(Row row, const int row_size);
        Mat(MatrixPtr pMat);
        Mat(int height, int width, int val = 0);
        Mat(const Size& size, int val = 0);
        Mat(L l);
        Mat(LL ll);
        virtual ~Mat();

        Row& operator[](int idx);

        virtual bool IsGood() const;

        virtual Mat<T> Mult(const Mat<T>& mat) const;
        virtual Mat<T>& Transpose();
        virtual Mat<T> Copy() const;
        virtual Mat<T>& Print();
        virtual Mat<T> CopyROI(Rect<int> roi) const;

        Size size() const { return mMat ? Size(mMat->col_count, mMat->row_count) : Size(0, 0); }
        inline T getAt(int i, int j) const { return mMat->values[i][j]; }
        inline void setAt(int i, int j, T val) { mMat->values[i][j] = val; }

        /* - - - ROW COL Setters/Getters - - - */
        // return row must be deleted
        Row row(int i) const;

        // return col must be deleted
        Col col(int j) const; // not ref since cannot modify original column..

        // param row must be deleted
        void row(int i, Row row);

        // param col must be deleted
        void col(int j, Col col);
        /* - - - - - - - - - - - - - - - - - - */

        /* ROW COL PUSH / POP */ /// As usual, params must be deleted since they are copied (for consistency)
        void pushCol(Col col);
        void pushRow(Row row);

    protected:
        Mat(Matrix_T<T>* mat);

    protected:
        MatrixPtr mMat;
    };
}

/// / / / / / /// / / / / / /// / / / / / /// / / / / /


namespace ml {

    // dot, sum, diff, absdiff, sqr, sqrt of T* array

    template <typename T>
    T dot(const T* v1, const T* v2, const size_t size) {
        T sum = T(0.0);
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < size; ++i)
            sum = sum + (v1[i] * v2[i]);
        return sum;
    }

    template <typename T>
    T* add(const T* v1, const T* v2, const size_t size) {
        assert(v1 && v2);
        T* v3 = new T[size];
        #pragma omp parallel for
        for (int i = 0; i < size; ++i)
            v3[i] = v1[i] + v2[i];
        return v3;
    }

    template <typename T>
    T* sum(const T* v1, const size_t size) {
        assert(v1 && size > 0);
        T _sum = 0;
        #pragma omp parallel for reduction(+:_sum)
        for (int i = 0; i < size; ++i)
            _sum = _sum + v1[i];
        return _sum;
    }

    template <typename T>
    T* sumVec(const std::vector<T*> v, const size_t size) {
        T* input = NULL;
        if (size && v.size() > 0) {
            input = v[0];
            T* r = NULL;
            for (int i = 1; i < v.size(); ++i) {
                r = ml::sum<T>(input, v[i], size);
                if (i > 1) delete[] input;
                input = r;
            }
        }
        return input;
    }

    template <typename T>
    T* diff(const T* v1, const T* v2, const size_t size) {
        assert(v1 && v2);
        T* v3 = new T[size];
        #pragma omp parallel for
        for (int i = 0; i < size; ++i)
            v3[i] = v1[i] - v2[i];
        return v3;
    }

    template <typename T>
    T* absdiff(const T* v1, const T* v2, const size_t size) {
        assert(v1 && v2);
        T* v3 = new T[size];
        #pragma omp parallel for
        for (int i = 0; i < size; ++i)
            v3[i] = abs(v1[i] - v2[i]);
        return v3;
    }

    template <typename T>
    T* sqr(const T* v1, const size_t size) {
        assert(v1);
        T* v3 = new T[size];
        #pragma omp parallel for
        for (int i = 0; i < size; ++i)
            v3[i] = v1[i] * v1[i];
        return v3;
    }

    template <typename T>
    T* Sqrt(const T* v1, const size_t size) {
        assert(v1);
        T* v3 = new T[size];
        #pragma omp parallel for
        for (int i = 0; i < size; ++i)
            v3[i] = sqrt(v1[i]);
        return v3;
    }


    /* Mat<T> Specific operations */

    /// element-wise sum m1 and m2 elements. the mats must be the same size
    template <typename T>
    ml::Mat<T> Sum(const ml::Mat<T>& m1, const ml::Mat<T>& m2) {
        typedef Mat<T> Mat;
        assert(m1.size() == m2.size());
        Mat m3(m1.size(), 0);
        for (int i = 0; i < m1.size().cy; ++i) {
            for (int j = 0; j < m2.size().cx; ++j) {
                m3.setAt(i, j, m1.getAt(i, j) + m2.getAt(i, j));
            }
        }
        return m3;
    }

    /// sums elementwise each row, stores into a mat with one row (contains sum of rows)
    template <typename T>
    ml::Mat<T> SumRows(const ml::Mat<T>& mat) {
        if (mat.size() == ml::Size(0, 0))
            return ml::Mat<T>();

        typedef Mat<T> Mat;
        Mat m3(1, mat.size().cx);
        typename ml::Mat<T>::Row v = mat.row(0);
        for (int i = 1; i < mat.size().cy; ++i) {
            T* t = ml::add<T>(v, mat.row(i), mat.size().cx);
            if (i > 1) delete[] v;
            v = t;
        }

        m3.row(0, v);
        delete[] v;
        return m3;
    }

    // sumrows but with a bunch of mats
    template <typename T>
    ml::Mat<T> SumRows(const std::vector<ml::Mat<T>>& mats) {
        typedef Mat<T> Mat;

        // determine the number of columns
        int cx = 0;
        for (int i = 0; i < mats.size(); ++i) {
            if (mats[i].size().cx != 0) {
                cx = mats[i].size().cx;
                break;
            }
        }
        if (cx == 0)
            return ml::Mat<T>();

        Mat m3(mats.size(), cx, 0);
        #pragma omp parallel for
        for (int i = 0; i < mats.size(); ++i) {
            if (mats[i].size().cx > 0) {
                T* rowCopy = SumRows<T>(mats[i]).row(0);
                m3.row(i, rowCopy);
                delete[] rowCopy;
            }
        }
        return SumRows<T>(m3);
    }

    template <typename T>
    ml::Mat<T> AbsDiff(const ml::Mat<T>& m1, const ml::Mat<T>& m2) {
        typedef Mat<T> Mat;
        assert(m1.size() == m2.size());
        Mat m3(m1.size(), 0);
        for (int i = 0; i < m1.size().cy; ++i) {
            for (int j = 0; j < m2.size().cx; ++j) {
                m3.setAt(i, j, abs(m1.getAt(i, j) - m2.getAt(i, j)));
            }
        }
        return m3;
    }

    // m1 - m2 -> m3
    template <typename T>
    ml::Mat<T> Diff(const ml::Mat<T>& m1, const ml::Mat<T>& m2) {
        typedef Mat<T> Mat;
        assert(m1.size() == m2.size());
        Mat m3(m1.size(), 0);
        for (int i = 0; i < m1.size().cy; ++i) {
            for (int j = 0; j < m2.size().cx; ++j) {
                m3.setAt(i, j, m1.getAt(i, j) - m2.getAt(i, j));
            }
        }
        return m3;
    }

    // 1 - m1 -> m3
    template <typename T>
    ml::Mat<T> Diff(T val, const ml::Mat<T>& m1) {
        typedef Mat<T> Mat;
        Mat m3(m1.size(), 0);
        for (int i = 0; i < m1.size().cy; ++i) {
            for (int j = 0; j < m1.size().cx; ++j) {
                m3.setAt(i, j, val - m1.getAt(i, j));
            }
        }
        return m3;
    }

    // m1 - 1 -> m3
    template <typename T>
    ml::Mat<T> Diff(const ml::Mat<T>& m1, T val) {
        typedef Mat<T> Mat;
        Mat m3(m1.size(), 0);
        for (int i = 0; i < m1.size().cy; ++i) {
            for (int j = 0; j < m1.size().cx; ++j) {
                m3.setAt(i, j, m1.getAt(i, j) - val);
            }
        }
        return m3;
    }


    /* Mult */
    template <typename T>
    ml::Mat<T> Mult(const ml::Mat<T>& m1, const ml::Mat<T>& m2, bool bIsTransposedAlready=false) {
        typedef ml::Mat<T> Mat;
        Mat m2Copy = m2.Copy();
        if (!bIsTransposedAlready)
            m2Copy.Transpose();
        Mat res(m1.size().cy, m2Copy.size().cy);
        //#pragma omp parallel for
        for (int i = 0; i < m1.size().cy; i++) {
            typename Mat::Row r = m1.row(i);
            const size_t newRowSize = m2Copy.size().cy;
            typename Mat::Row newRow = new T[newRowSize];
            for (size_t j = 0; j < newRowSize; j++) {
                typename Mat::Col c = m2Copy.row(j); // accessable like a row since transposed
                newRow[j] = dot<T>(r, c, m2Copy.size().cx);
                delete[] c;
            }
            res.row(i, newRow);
            delete[] r;
            delete[] newRow;
        }
        return res;
    }

    // Element-wise multiply
    template <typename T>
    ml::Mat<T> ElementMult(const ml::Mat<T>& m1, const ml::Mat<T>& m2) {
        assert(m1.size() == m2.size());
        typedef ml::Mat<T> Mat;
        Mat m3(m1.size(), 0);
        for (int i = 0; i < m1.size().cy; ++i) {
            for (int j = 0; j < m1.size().cx; ++j) {
                m3.setAt(i, j, m1.getAt(i,j) + m2.getAt(i,j));
            }
        }
        return m3; 
    }


    /* determinant */
    template <typename T>
    T det(const Mat<T>& A);

    // MATRIX M is the matrix that is disconnected
    // from the x_iter and y_iter column and row
    // (where the scaler exists)
    template <typename T>
    Mat<T> findM(const Mat<T>& A, int i, int j, int SIZE);
}


/// TODO move out into another file
namespace ml {
    template <typename T>
    ml::Mat<T> initWeightsNormalDist(int rows, int cols, T mean=0.001f, T stddev=0.0001f) {
        ml::Mat<T> weights(rows, cols, 0);
		if (stddev == 0)
			return weights;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(mean, stddev);

        for (int i = 0; i < weights.size().cy; ++i) {
            for (int j = 0; j < weights.size().cx; ++j) {
                weights.setAt(i, j, std::round(d(gen)));
            }
        }

        return weights;
    }
}