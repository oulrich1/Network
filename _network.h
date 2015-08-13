#pragma once

#include "math/rect.h"
#include "Matrix/matrix.h"

using namespace std;

////
namespace ml {
	template <typename T>
	class ILayer {
	public:
		typedef ml::Matrix<T> Mat;
		typedef std::function<T(T)> Sigmoid;

	public:
		ILayer() { mNumInputNodes = 0; mNumOutputNodes = 0; }
		virtual ~ILayer() { }

	public: // overridables
		virtual void initWeights(size_t numInputNodes = 0, bool bIsPass = false) = 0; // ..

		virtual Vector<T> feed(Vector<T> in) = 0;
		virtual Vector<T> activate(Vector<T> in) = 0;
		virtual void activate(int nodeIdx, T in) = 0;

		// protected:

		// sizing properties
		static size_t GetNumBiasNodes() { return 1; }
		virtual void setNumInputNodes(size_t nInputNodes)
		{
			mNumInputNodes = nInputNodes;
			mNumOutputNodes = (mNumInputNodes + GetNumBiasNodes());
		}
		virtual size_t getNumInputNodes() const { return mNumInputNodes; }
		virtual void setNumOutputNodes(size_t nOutputNodes)
		{
			mNumOutputNodes = nOutputNodes;
			mNumInputNodes = (mNumOutputNodes - GetNumBiasNodes());
		}
		virtual size_t getNumOutputNodes() const { return mNumOutputNodes; }
		virtual size_t getOutputSize() const { return getNumOutputNodes(); }
		virtual size_t getInputSize() const { return getNumInputNodes(); }

		virtual inline Vector<T> getOutput() const { return mOutput; }

	protected:
		inline void setOutput(Vector<T> v) { mOutput = v; }

	protected:
		Vector<T> mOutput; // for getting output after feed or activate
		size_t mNumInputNodes;  // the num inputs into this layer
		size_t mNumOutputNodes; // the num outputs out of this layer (usually just mInputLayerSize + #biases)
	};


	template <typename T>
	class Layer : public ILayer<T> {
	public:
		typedef ILayer<T> baseclass;
		typedef typename baseclass::Mat Mat;
		typedef typename baseclass::Sigmoid Sigmoid;
		Sigmoid sigmoid;

	public:
		Layer(size_t inputSize);
		virtual ~Layer();

		virtual void initWeights(size_t numInputNodes = 0, bool bIsPass = false);

		// overrides
		virtual Vector<T> feed(Vector<T> in);
		virtual Vector<T> activate(Vector<T> in);
		virtual void activate(int nodeIdx, T in);
		virtual size_t getOutputSize() const
		{
			return baseclass::getNumOutputNodes();
		}
		virtual size_t getInputSize() const
		{
			return baseclass::getNumInputNodes();
		}

	public:
		virtual void setIsFeedPassThrough(bool bPass)
		{
			bIsFeedPassThrough = bPass;
		}

	private:
		Mat weights;
		bool bIsFeedPassThrough = false;
	};


	template <typename T>
	Layer<T>::Layer(size_t inputSize) {
		baseclass::setNumInputNodes(inputSize);
		if (inputSize) {
			baseclass::setOutput(ml::Vector<T>(inputSize
				+ baseclass::GetNumBiasNodes()));
		}
	}

	template <typename T>
	Layer<T>::~Layer() {
	}

	/* weighs the input and activates the nodes */
	template <typename T>
	Vector<T> Layer<T>::feed(Vector<T> in) {
		if (bIsFeedPassThrough || !weights.IsGood())
			return in; //
		ml::Size weightsSize = weights.size();
#pragma omp parallel for
		for (int i = 0; i < weightsSize.cy; ++i) {
			activate(i, in.Dot(weights.at(i)));
		}
		return baseclass::getOutput();
	}

	/* activates the nodes' outputs */
	template <typename T>
	Vector<T> Layer<T>::activate(Vector<T> in) {
		for (size_t i = 0; i < in.size(); i++)
			activate(i, in.at(i));
		return baseclass::getOutput();
	}

	/* activates ONE node output */
	template <typename T>
	void Layer<T>::activate(int nodeIdx, T in) {
		// TODO: parallelize sigmoid
		T val = sigmoid(in);
		baseclass::getOutput().set(nodeIdx, val);
	}

	/* param - if bIsPass == true then this layer is a pass through and the input
	  can simply forward without being multiplied - Kludgey.. */
	template <typename T>
	void Layer<T>::initWeights(size_t numInputsIncoming, bool bIsPass/*=false*/) {
		if (bIsPass) {
			setIsFeedPassThrough(true);
			return;
		} else {
			setIsFeedPassThrough(false);
		}
		// note: transposed for fast math
		weights = Mat(baseclass::getInputSize(), numInputsIncoming, 1);
	}


	////

	template <typename T>
	class Network;

	// SubnetLayer - Wraps a subnet into a layer.. looks like a
	// layer, functions like a layer.. actually subnet.
	template <typename T>
	class SubnetLayer : public Layer<T>
	{
	public:
		typedef Layer<T> baseclass;
		typedef typename Layer<T>::Mat Mat;

	public:
		SubnetLayer(Network<T>* subnet) {
			_subnet = subnet;
		}
		virtual ~SubnetLayer() {
			if (_subnet) {
				delete _subnet;
			}
		}
		virtual Vector<T> feed(Vector<T> in) override {
			if (!_subnet)
				return baseclass::getOutput();
			setOutputs(_subnet->feed(in));
			return baseclass::getOutput();
		}
	private:
		Network<T>* _subnet;
	};


	////

	template <typename T>
	class Network : public ILayer<T>
	{
	public:
		typedef ILayer<T> baseclass;
		typedef ml::Matrix<T> Mat;

	public:
		Network();
		virtual ~Network();
		void push_layer(ILayer<T>* layer);
		virtual void initWeights(size_t numInputNodes = 0, bool bIsPass = false);
		Vector<T> feedNetwork(Vector<T> in, bool bActivateFirstLayer = false);
		Vector<T> feedNetwork(); // input comes from the irst layer's outputs
		Vector<T> activateFirstLayer(Vector<T> in);
		void activateFirstLayer(int idx, T in);

	public: // Override ILayer
		virtual Vector<T> feed(Vector<T> in) override;
		virtual Vector<T> activate(Vector<T> in) override;
		virtual void activate(int nodeIdx, T in) override;
		virtual size_t getOutputSize() const
		{
			const size_t size = layers.size(); // iff 0 then invalid
			return size > 0 ? layers[size - 1]->getOutputSize() : 0;
		}
		virtual size_t getInputSize() const
		{
			const size_t size = layers.size(); // iff 0 then invalid
			return size > 0 ? layers[size - 1]->getInputSize() : 0;
		}

	protected:

	private:
		std::vector<ILayer<T>*> layers;
		std::map<ILayer<T>*, ILayer<T>*> layerMap;
	};

	// virtual overrides of ILayer
	template <typename T>
	Vector<T> Network<T>::feed(Vector<T> in) {
		return feedNetwork(in);
	}

	template <typename T>
	Vector<T> Network<T>::activate(Vector<T> in) {
		return activateFirstLayer(in);
	}

	template <typename T>
	void Network<T>::activate(int nodeIdx, T in) {
		activateFirstLayer(nodeIdx, in);
	}
	// end ILayer Overrides


	// constructor
	template <typename T>
	Network<T>::Network() : baseclass() {
	}

	template <typename T>
	Network<T>::~Network() {
		for (auto it = layerMap.begin(); it != layerMap.end(); ++it) {
			if (it->first && it->second)
				delete it->second;
			layerMap.erase(it);
		}
		layers.clear();
	}

	// Implementation
	template <typename T>
	void Network<T>::push_layer(ILayer<T>* layer) {
		if (layers.size() == 0) // pushing first layer..
			baseclass::setNumInputNodes(layer->getInputSize());
		layers.push_back(layer);
		auto it = layerMap.find(layer);
		if (it == layerMap.end())
			layerMap[layer] = layer;
	}

	/* - - - - - - - - - - - - - - - - - - - - - - - - -
	  Usage:
	  network.push(layers);
	  network->feed(input);
	  */
	template <typename T>
	Vector<T> Network<T>::feedNetwork(Vector<T> in,
		bool bActivateFirstLayer/*=false*/)
	{
		if (layers.size() <= 0)
			return in; // no network
		// figure out the transformed input, does the first layer
		// contribute to the activation or is it simply a placeholder
		// for the input.. since the coefficients are what really matter..
		Vector<T> next_in;
		if (bActivateFirstLayer) {
			next_in = activateFirstLayer(in);
		} else {
			next_in = in;
		}

		for (size_t i = 0; i < layers.size(); ++i) {
			next_in = layers[i]->feed(next_in);
		}
		return next_in.CopyROI(Range<int>(0, next_in.size() - 1));
	}

	/* - - - - - - - - - - - - - - - - - - - - - - - - -
	  Usage:
	  layers[0]->activate(in);
	  network.push(layers);
	  network->feed();
	  */
	template <typename T>
	Vector<T> Network<T>::feedNetwork() {
		// feed forward, using the first layer's output
		return feed(layers[0]->getOutput());
	}

	template <typename T>
	Vector<T> Network<T>::activateFirstLayer(Vector<T> in)
	{
		if (layers.size() <= 0)
			return in;
		return layers[0]->activate(in);
	}

	template <typename T>
	void Network<T>::activateFirstLayer(int idx, T in)
	{
		if (layers.size() <= 0)
			return;
		layers[0]->activate(idx, in);
	}

	/* - - - - - - - - - - - - - - - - - - - - - - - - -  >>
	// protected

	/* @param numInputNodes - 0 if weights for the
	first layer dont matter, as in the input layer,
	just here to implement the same interface..  */
	template <typename T>
	void Network<T>::initWeights
		(
		size_t numIncomingNodes/*=0*/,
		bool bIsFeedPassThrough/*=false*/
		)
	{
		const size_t numLayers = layers.size();
		if (numLayers == 0)
			return;

		/* Set this network's num output nodes to the numOutput
		   nodes of the last layer */
		const size_t numOutputNodes = layers[numLayers - 1]->getOutputSize();
		baseclass::setNumOutputNodes(numOutputNodes);

		/* Initialize weights based on incoming nodes from the
		   previous layer and the target nodes in the next layer */
		for (size_t i = 0; i < numLayers; ++i) {
			if (i) {
				numIncomingNodes = layers[i - 1]->getOutputSize();
				bIsFeedPassThrough = false;
			} else {
				bIsFeedPassThrough = true;
			}
			layers[i]->initWeights(numIncomingNodes, bIsFeedPassThrough);
		}
	}

} // namespace ml
