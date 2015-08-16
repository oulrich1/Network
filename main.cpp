
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
	ml::Mat<T> input(1, 100, 1);
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

void test_crazy_network_1() {
    using namespace std;
    using namespace ml;
    typedef int T;

    Timer<float> timer;
    timer.start();
    

    // Defining network
    Network<T>* network = new Network<T>();
    ILayer<T>* l1 = new Layer<T>(100);
    ILayer<T>* l2 = new Layer<T>(200);
    ILayer<T>* l3 = new Layer<T>(500);

    // Defining subnet, middle node is actually l2. sorta recurrent
    Network<T>* aSubNet = new Network<T>();//.. = something else
    {
        ILayer<T>* s1 = new Layer<T>(100);
        ILayer<T>* s3 = new Layer<T>(5000);
        aSubNet->setInputLayer(s1);
        aSubNet->connect(s1, l2);
        aSubNet->connect(l2, s3);
        aSubNet->setOutputLayer(s3);
        aSubNet->setName("SubNet 1");
    }

    // Defining rest of network
    ILayer<T>* l4 = aSubNet;
    ILayer<T>* l5 = new Layer<T>(2);
    l1->setName("L1");
    l2->setName("L2");
    l3->setName("L3");
    l4->setName("L4");
    l5->setName("L5");
    network->setInputLayer(l1);
    network->connect(l1, l2);
    network->connect(l2, l3);
    network->connect(l3, l4);
    network->connect(l3, l2); // note: circularity back to l2
    network->connect(l4, l2); //       here again as well
    network->connect(l4, l5); // note: connecting subnet arbitrarily to another layer..
    network->setOutputLayer(l5);
    network->init();
    ml::Mat<T> input(1, 100, 1); // input vec must be the same size as the input layer's size
    network->feed(input);

    ml::Mat<T> samples(100, 100, 1);
    network->train(samples);

    timer.stop();
    cout << timer.getTime() << endl;
}

int main() {
	//test1();
	test2();
	test3();
    test_crazy_network_1();
	return 0;
}

//// ----------------------------------------------------------------- ////
