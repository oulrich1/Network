
#include "allheader.h"

#include <stdio.h>
#include <string.h>
#include <cassert>


void test_nn_math() {
    BEGIN_TESTS("Testing ml::FastMath / SSE instruction");
    float t = 0;      // time
    float result = 0; // common float result

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    __m128 Special = _mm_set1_ps(2);
    __m128 Identity = _mm_set1_ps(1.);

    __m128 res = sse_dot4(Special, Identity);
    _mm_store_ss(&result, res);

    std::cout << result << std::endl;
#endif

    std::vector<float> v1 = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    t = TIME_ASSIGN(result, dot<float>(v1, v1));

    std::cout << result << " in time: " << t << std::endl;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    __m128 res128;
    typedef std::vector<__m128> vec_mm;
    vec_mm v = make_mm_vec<vec_mm>(_mm_set1_ps(1), 7);
    t = TIME_ASSIGN(res128, sse_dot_vec(v, v));

    _mm_store_ss(&result, res128);
    std::cout << result << " in time: " << t << std::endl;
#else
    std::cout << "SSE tests skipped on non-x86 architecture" << std::endl;
#endif
    std::cout << std::endl << ">> Done with 'Timer' testing." << std::endl;
    std::cout << std::endl;
}

void test_rect() {
    BEGIN_TESTS("Testing ml::Rect");
  using namespace std;
  typedef double T;
  {
    Rect<T> r1(-1, -1, 1, 1);
    Rect<T> r2(0, 0, 2, 2);
    bool bDoesIntersect = r1.intersects(r2);
    assert(bDoesIntersect);

    // print the rects
    cout << "r1 = Rect " << r1 << endl;
    cout << "r2 = Rect " << r2 << endl;
    cout << "r1.intersects(r2) == " << r1.intersects(r2) << endl;
    cout << "r1.get_intersect(r2) == " << r1.get_intersect(r2) << endl;
    cout << "r1 + r2 == " << r1 + r2 << endl;
  }

  cout << endl;
  {
    Rect<T> r1(-1123523.23, -0.0000101, 5234234, 0.0000101);
    Rect<T> r2(-0.0000101, -12342342, 0.0000101, 234235);
    // print the rects
    cout << "r1 = Rect " << r1 << endl;
    cout << "r2 = Rect " << r2 << endl;
    cout << "r1.intersects(r2) == " << r1.intersects(r2) << endl;
    cout << "r1.get_intersect(r2) == " << r1.get_intersect(r2) << endl;
    cout << "r1 + r2 == " << r1 + r2 << endl;
  }

  cout << endl << ">> Done with Rect Testing." << endl;
    cout << endl;
}


void test_matrix2 () {
    BEGIN_TESTS("Testing Updated Mat<T> Wrapper");
    using namespace ml;
    using namespace std;
    typedef float T;
    Mat<T> m1(100, 100, 1);
    Mat<T> m2(100, 10, 1);
    Mat<T> m3 = m1.Mult(m2);
    Mat<T> f = ml::Mult<T>(m1, m2);
    Mat<T> x{ { 1,2 },{ 2,3 } };
    Timer<float> timer;
    timer.start();
    T d;
    d = ml::det<T>(x);
    timer.stop();
    cout << ">>  det= " << d << endl;
    cout << ">> Took Time: " << timer.getTime() << endl;
}


void test_refs() {
    BEGIN_TESTS("Testing Reference Counted Arrays..");
    using namespace std;

    int* pArr = new int[100];
    for (size_t i = 0; i < 100; i++) {
        pArr[i] = 1;
    }

    ml::refArr<int> r2;
    {
        ml::refArr<int> r1(pArr, 100);
        cout << "r1 #refs: " << r1.numReferences() << endl;
        cout << "r2 #refs: " << r2.numReferences() << endl;

        r2 = r1;

        cout << "r1 #refs: " << r1.numReferences() << endl;
        cout << "r2 #refs: " << r2.numReferences() << endl;

        for (size_t i = 0; i < 100; i++) {
            cout << r1.data()[i] << " ";
        }
        cout << "r1 #refs: " << r1.numReferences() << endl;
        cout << "r2 #refs: " << r2.numReferences() << endl;
        cout << endl;
        cout << endl;

        for (size_t i = 0; i < 100; i++) {
            cout << r2.data()[i] << " ";
        }
        cout << "r1 #refs: " << r1.numReferences() << endl;
        cout << "r2 #refs: " << r2.numReferences() << endl;
        cout << endl;
        cout << endl;
    }

    for (size_t i = 0; i < 100; i++) {
        cout << r2[i] << " ";
    }
    cout << "r2 #refs: " << r2.numReferences() << endl;
    cout << endl;
    cout << endl;

    ml::refArr<double> r1;
    ml::refArr<double> r3;
    cout << "r1 #refs: " << r1.numReferences() << endl;
    cout << "r2 #refs: " << r2.numReferences() << endl;
    cout << "r3 #refs: " << r3.numReferences() << endl;
    cout << endl;

    cout << "Releasing r2" << endl;
    int* data = r2.release();

    cout << "Deleting r2's old data" << endl;
    delete[] data;

    cout << endl;
    cout << "r1 #refs: " << r1.numReferences() << endl;
    cout << "r2 #refs: " << r2.numReferences() << endl;
    cout << "r3 #refs: " << r3.numReferences() << endl;
    cout << endl;
    cout << endl;

    BEGIN_TESTS("Testing Referenced Counted Arrays (2)")
    {
        ml::refArr<double> r5;
        ml::refArr<double> r6;
        ml::refArr<double> r7;
        ml::refArr<double> r8;
        r8 = new double[20];
        for (size_t i = 0; i < 20; i++)
            r8[i] = 2;
        r8.setSize(20);
        r8.print();
        r6 = 0;
        r7 = r8;
        r5 = 0;
        cout << "r8 == r7: " << (r8 == r7) << endl;
        cout << "r8 == r6: " << (r8 == r6) << endl;
        cout << "r5 == r6: " << (r5 == r6) << endl;
        assert((r8 == r7) == true);
        assert((r8 == r6) == false);
        assert((r5 == r6) == true);
    }

    {
        ml::refArr<double> r6(new double[10000], 10000);
        ml::refArr<double> r7;
        r7 = r6;
        ml::refArr<double> r8(r7);
        ml::refArr<double> r9(r6);

        cout << "r6(new double[12342134123])" << endl;
        cout << "r7 = r6" << endl;
        cout << "ref<T> r8(r7); " << endl;
        cout << "ref<T> r9(r7); " << endl;
        cout << "r9 refs: " << r9.numReferences() << endl;
        cout << "r7.release(): " << endl;
        r7.release();
        cout << "r9 refs: " << r9.numReferences() << endl;
        cout << "r6.release(): " << endl;
        r6.release();
        cout << "r9 refs: " << r9.numReferences() << endl;
        cout << "r8.release(): " << endl;
        r8.release();
        cout << "r9 refs: " << r9.numReferences() << endl;
        cout << "r9 = 0;" << endl;
        r9 = 0;
        cout << "r9 refs: " << r9.numReferences() << endl;
    }
}

void refInit(ml::refArr<double> a, double val) {
    printf("'a' refs: %d\n", a.numReferences());
    for (size_t i = 0; i < a.getSize(); i++)
        a[i] = val;
}

void test_refs2() {
    ml::refArr<double> r6(new double[100], 100);
    {
        r6.print();
        ml::refArr<double> r8(r6);
        ml::refArr<double> r9(r6);
        printf("'r6' refs: %d\n", r6.numReferences());
        refInit(r8, 2);
        printf("'r6' refs: %d\n", r6.numReferences());
        r6.print();
    }
    printf("'r6' refs: %d\n", r6.numReferences());
    r6.print();
}

int main(int argc, char const *argv[])
{
    using namespace std;
    using namespace Utility;

    print(">> Testing..");
    test_nn_math();
    test_rect();
    //test_matrix();
    test_matrix2();
    test_refs();
    test_refs2();
    print(">> All Done.");
    return 0;
}
