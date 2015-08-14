
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
	aSubNet->connect(aSubNet->getInputLayer(), new Layer<T>(1000));
	aSubNet->setOutputLayer(aSubNet->getInputLayer()->siblings[0]);
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
	network->setOutputLayer(l4);
	network->init();
	ml::Mat<T> input(100, 100, 1);
	network->feed(input);

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
	test3();
	return 0;
}

//// ----------------------------------------------------------------- ////
