
#include <stdlib.h>
#include <iostream>
#include "allheader.h"
#include "network.h"

template <typename T>
T* newArray(int nSize, T val = 0) {
	T* arr = new T[nSize];
	#pragma omp parallel for
	for (int i = 0; i < nSize; i++)
		arr[i] = val;
	return arr;
}


//// ----------------------------------------------------------------- ////


//void test1() {
//	using namespace std;
//	typedef float T;
//	typedef ml::Layer<T> Layer;
//	typedef ml::Network<T> Network;
//	typedef Network::Mat Mat;
//	typedef ml::Vector<T> Vec;
//
//	Timer<float> timer;
//
//	BEGIN_TESTS("Creating and Multiplying RefArr<float> Matricies (seconds)")
//		timer.start();
//	int nRows = 100;
//	int nCols = 1024;
//	std::vector<ml::refArr<T>> mat;
//	for (int i = 0; i < nRows; i++) {
//		ml::refArr<T> arr(newArray<T>(nCols, 1), nCols);
//		mat.push_back(arr);
//	}
//	timer.stop();
//	cout << "  INIT Mat1: " << timer.getTime() << endl;
//
//	timer.start();
//	int nRows2 = nCols;
//	int nCols2 = nRows;
//	std::vector<ml::refArr<T>> mat2;
//	for (int i = 0; i < nRows2; i++) {
//		ml::refArr<T> arr(newArray<T>(nCols2, 1), nCols2);
//		mat2.push_back(arr);
//	}
//	timer.stop();
//	cout << "  INIT Mat1: " << timer.getTime() << endl;
//
//	timer.start();
//	int rows = nRows;
//	int cols = 100;
//	std::vector<ml::refArr<T>> result;
//	for (int i = 0; i < rows; i++) {
//		ml::refArr<T> arr(newArray<T>(rows, 42), rows);
//		result.push_back(arr); // this should change on assign
//	}
//	timer.stop();
//	cout << "  INIT Result: " << timer.getTime() << endl;
//
//	timer.start();
//	for (int i = 0; i < nRows; i++) {
//		for (size_t j = 0; j < cols; j++) {
//			for (size_t k = 0; k < nCols; k++) {
//				result[i][j] += mat[i][k] * mat2[k][j];
//			}
//		}
//	}
//	timer.stop();
//	cout << "  Result: " << timer.getTime() << endl;
//
//
//	BEGIN_TESTS("Creating and Multiplying float** Matricies (seconds)")
//
//	timer.start();
//	T** aa = new T*[nRows];
//	for (size_t i = 0; i < nRows; i++) {
//		aa[i] = newArray<T>(nCols, 1);
//	}
//	timer.stop();
//	cout << "  AA init: " << timer.getTime() << endl;
//
//	timer.start();
//	T** bb = new T*[nCols];
//	for (size_t i = 0; i < nCols; i++) {
//		bb[i] = newArray<T>(nRows, 1);
//	}
//	timer.stop();
//	cout << "  BB init: " << timer.getTime() << endl;
//
//	T** cc = new T*[nRows];
//	for (size_t i = 0; i < nRows; i++) {
//		cc[i] = newArray<T>(cols, 1);
//	}
//
//	timer.start();
//	for (int i = 0; i < nRows; i++) {
//		for (size_t j = 0; j < cols; j++) {
//			for (size_t k = 0; k < nCols; k++) {
//				result[i][j] += aa[i][k] * bb[k][j];
//			}
//		}
//	}
//	timer.stop();
//	cout << "  CC result: " << timer.getTime() << endl;
//
//
//	BEGIN_TESTS("Creating and Multiplying Matrix_T<T> Matricies (seconds)")
//
//	ml::Matrix_T<T>* mm1 = ml::matrix_create<T>(nRows, nCols);
//	ml::Matrix_T<T>* mm2 = ml::matrix_create<T>(nCols, nRows);
//	timer.start();
//	ml::Matrix_T<T>* mmRes = ml::matrix_mult_ijk<T>(mm1, mm2);
//	timer.stop();
//	cout << "  ml::Matrix_T<T>: " << timer.getTime() << endl;
//
//
//	BEGIN_TESTS("Creating and Multiplying Matrix<T> Networks (seconds)")
//		// ml::matrix_print<T>(mmRes);
//
//		cout << ">> Loading Weights" << endl;
//
//}
//
//



//// ----------------------------------------------------------------- ////



void test2() {
	using namespace Utility;
	using namespace std;
	using namespace ml;
	typedef double T;

	ml::Mat<T> m = ml::initWeightsNormalDist<T>(5, 5, 0.01f, 2);
	m.Print();

	Timer<float> timer;
	timer.start();
	Network<T>* network = new Network<T>();
	ILayer<T>* l1 = new Layer<T>(100);
	ILayer<T>* l2 = new Layer<T>(200);
	ILayer<T>* l3 = new Layer<T>(500);
	ILayer<T>* l4 = new Layer<T>(10);
	l1->setName("L1");
	l2->setName("L2");
	l3->setName("L3");
	l4->setName("L4");
	network->setInputLayer(l1);
	network->connect(l1, l2);
	network->connect(l2, l3);
	network->connect(l3, l4);
	network->setOutputLayer(l4);
	network->init();

  ml::Mat<T> input(100, 100, 1);
	network->feed(input);
	ml::Mat<T> output = network->getOutput();



	/*
	network->init();
	for sample in samples:
	 out = network->feed(sample);
	 // todo: back prop out into network
	*/


	timer.stop();
	cout << timer.getTime() << endl;
}

void test3() {
	using namespace std;
	using namespace ml;
	typedef int T;

	Timer<float> timer;
	timer.start();
	Network<T>* aSubNet = new Network<T>();//.. = something else
	aSubNet->setInputLayer(new Layer<T>(50));
	aSubNet->setOutputLayer(aSubNet->getOutputLayer());
	aSubNet->setName("SubNet 1");

	Network<T>* network = new Network<T>();
	ILayer<T>* l1 = new Layer<T>(100);
	ILayer<T>* l2 = new Layer<T>(200);
	ILayer<T>* l3 = new Layer<T>(500);
	ILayer<T>* l4 = aSubNet;
	l1->setName("L1");
	l2->setName("L2");
	l3->setName("L3");
	l4->setName("L4");
	network->setInputLayer(l1);
	network->connect(l1, l2);
	network->connect(l2, l3);
	network->connect(l3, l4);
	network->connect(l3, l2); // Sigmoid(l1 * W1 + l3 * W4) -> L2 output
	network->connect(l4, l2);
	network->init();
	network->setOutputLayer(l4);
	ml::Mat<T> input(100, 100, 1);
	network->feed(input);
	//network->getOutput(); // from l4

	/*
	network->init();
	for sample in samples:
	out = network->feed(sample);
	// todo: back prop out into network
	*/


	timer.stop();
	cout << timer.getTime() << endl;
}

int main() {
	//test1();
	test2();
	//test3();
	return 0;
}

//// ----------------------------------------------------------------- ////
