#include <iostream>
#include "allheader.h"
#include "network.h"

using namespace ml;
using namespace std;

void printMatrix(const Mat<double>& mat, const string& name) {
    cout << name << " (" << mat.size().cy << "x" << mat.size().cx << "):" << endl;
    for (int i = 0; i < mat.size().cy; i++) {
        cout << "  [ ";
        for (int j = 0; j < mat.size().cx; j++) {
            cout << mat.getAt(i, j) << " ";
        }
        cout << "]" << endl;
    }
}

int main() {
    cout << "=== HYPOTHESIS: Testing Mult function behavior ===" << endl;

    // Create test matrices
    Mat<double> A(1, 2, 0);
    A.setAt(0, 0, 1.0);
    A.setAt(0, 1, 2.0);

    Mat<double> B(2, 1, 0);
    B.setAt(0, 0, 3.0);
    B.setAt(1, 0, 4.0);

    printMatrix(A, "\nA");
    printMatrix(B, "B");

    cout << "\n--- Test 1: Mult(A, B, false) - Let Mult transpose B ---" << endl;
    Mat<double> result1 = Mult<double>(A, B, false);
    printMatrix(result1, "Result1");

    cout << "\n--- Test 2: Mult(A, B, true) - Tell Mult B is already transposed ---" << endl;
    Mat<double> result2 = Mult<double>(A, B, true);
    printMatrix(result2, "Result2");

    cout << "\n--- Test 3: Manually transpose B, then Mult(A, B_transposed, true) ---" << endl;
    Mat<double> B_copy = B.Copy();
    B_copy.Transpose();
    printMatrix(B_copy, "B transposed");
    Mat<double> result3 = Mult<double>(A, B_copy, true);
    printMatrix(result3, "Result3");

    cout << "\n--- Test 4: Use the underlying matrix_mult_ijk directly ---" << endl;
    Mat<double> A_for_ijk(1, 2, 0);
    A_for_ijk.setAt(0, 0, 1.0);
    A_for_ijk.setAt(0, 1, 2.0);

    Mat<double> B_for_ijk(2, 1, 0);
    B_for_ijk.setAt(0, 0, 3.0);
    B_for_ijk.setAt(1, 0, 4.0);

    printMatrix(A_for_ijk, "A for ijk");
    printMatrix(B_for_ijk, "B for ijk");

    // Use the built-in Mult method
    Mat<double> result4 = A_for_ijk.Mult(B_for_ijk);
    printMatrix(result4, "Result4 (using A.Mult(B))");

    return 0;
}
