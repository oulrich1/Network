#include <iostream>
#include <cmath>
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
    cout << "=== BUG HYPOTHESIS: Forward propagation Mult call ===" << endl;

    // Simulate what happens in forward propagation
    cout << "\n--- Simulating forward propagation ---" << endl;

    // Current layer has 2 nodes + 1 bias = 3 outputs
    Mat<double> activatedWithBias(1, 3, 0);
    activatedWithBias.setAt(0, 0, 0.5);
    activatedWithBias.setAt(0, 1, 0.7);
    activatedWithBias.setAt(0, 2, 1.0);  // bias

    // Next layer has 2 nodes
    // Weights are initialized as (numNodesNextLayer, numInputPerNode) = (2, 3)
    Mat<double> weights(2, 3, 0);
    weights.setAt(0, 0, 1.0);  // node 0: w0
    weights.setAt(0, 1, 2.0);  // node 0: w1
    weights.setAt(0, 2, 3.0);  // node 0: bias
    weights.setAt(1, 0, 4.0);  // node 1: w0
    weights.setAt(1, 1, 5.0);  // node 1: w1
    weights.setAt(1, 2, 6.0);  // node 1: bias

    printMatrix(activatedWithBias, "Activated with bias");
    printMatrix(weights, "Weights");

    // Manual calculation of expected output:
    // node 0: 0.5*1.0 + 0.7*2.0 + 1.0*3.0 = 0.5 + 1.4 + 3.0 = 4.9
    // node 1: 0.5*4.0 + 0.7*5.0 + 1.0*6.0 = 2.0 + 3.5 + 6.0 = 11.5
    cout << "\nExpected output: [4.9, 11.5]" << endl;

    cout << "\n--- Test 1: CURRENT CODE - Mult(activatedWithBias, weights, true) ---" << endl;
    Mat<double> output_bug = Mult<double>(activatedWithBias, weights, true);
    printMatrix(output_bug, "Output (BUG)");

    cout << "\n--- Test 2: FIXED CODE - Mult(activatedWithBias, weights, false) ---" << endl;
    Mat<double> output_fixed = Mult<double>(activatedWithBias, weights, false);
    printMatrix(output_fixed, "Output (FIXED)");

    // Verify
    bool bug_correct = (output_bug.size().cx == 2 && output_bug.size().cy == 1 &&
                       abs(output_bug.getAt(0, 0) - 4.9) < 0.01 &&
                       abs(output_bug.getAt(0, 1) - 11.5) < 0.01);

    bool fixed_correct = (output_fixed.size().cx == 2 && output_fixed.size().cy == 1 &&
                         abs(output_fixed.getAt(0, 0) - 4.9) < 0.01 &&
                         abs(output_fixed.getAt(0, 1) - 11.5) < 0.01);

    cout << "\n========================================" << endl;
    cout << "Bug version correct: " << (bug_correct ? "YES" : "NO") << endl;
    cout << "Fixed version correct: " << (fixed_correct ? "YES" : "NO") << endl;
    cout << "========================================" << endl;

    if (!bug_correct && fixed_correct) {
        cout << "\nCONCLUSION: BUG CONFIRMED! Line 413 should use 'false' not 'true'" << endl;
        return 0;
    } else if (bug_correct && !fixed_correct) {
        cout << "\nCONCLUSION: Current code is correct, hypothesis was wrong" << endl;
        return 1;
    } else {
        cout << "\nCONCLUSION: Inconclusive, need more investigation" << endl;
        return 2;
    }
}
