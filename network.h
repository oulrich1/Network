#pragma once

#include <stack>
#include <vector>
#include <algorithm>
#include "math/rect.h"
#include "Matrix/matrix.h"


using namespace std;


#ifndef BIAS_UNIT_VAL
#define BIAS_UNIT_VAL 1
#endif


namespace ml {
    template <typename T>
    T* makeBias(size_t els, T val) {
        if (els == 0) return NULL;
        T* bias = new T[els];
        for (size_t i = 0; i < els; i++) {
          bias[i] = val;
        }
        return bias;
    }

    template <typename T>
    void pushBiasCol(ml::Mat<T>& mat, int bias = BIAS_UNIT_VAL) {
        if (!mat.IsGood()) return;
        T* col = makeBias<T>(mat.size().cy, bias);
        mat.pushCol(col);
        delete[] col;
    }
}


////
namespace ml {

    using namespace std;
    using namespace StackEx;


    template <typename T>
    class INode {

    public:
        INode() { bIsVisited = false; }
        virtual ~INode() {  }
        bool IsVisited() const { return bIsVisited; }
        void SetVisited(bool b) { bIsVisited = b; }
        bool IsAboutToBeVisited() { return bIsAboutToBeVisited; }
        void SetAboutToBeVisited(bool b) { bIsAboutToBeVisited = b;  }

    private:
        bool bIsVisited;
        bool bIsAboutToBeVisited;
    };


    /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    */

    template <typename T>
    class ILayer : public INode<T> {
    public:
        typedef ml::Mat<T> Mat;
        typedef std::function<T(T)> Sigmoid;
        typedef std::vector<ILayer<T>*> Siblings;

    public:
        ILayer() : INode<T>() { mNumInputNodes = 0; mNumOutputNodes = 0; }
        virtual ~ILayer() { siblings.clear(); }

    public: // ILayer<T> overrides
        virtual void init(); // ..

         // feeding is weighing and activating across ALL edges from this sib to it's sib
        virtual void feed(ml::Mat<T> in, int epoch) = 0;
        virtual typename ml::Mat<T>::Row activate(typename ml::Mat<T>::Row in) = 0;
        virtual void activate(int nodeIdx, T in) = 0;
        virtual void initWeights(ILayer<T>* pSib);
        virtual ml::Mat<T> getActivatedInput() = 0;
        virtual void setActivatedInput(ml::Mat<T> activatedInput) = 0;

    public:
        virtual void connect(ILayer<T>* nextLayer);
        virtual void connect(std::vector<ILayer<T>*> nextLayers);
        virtual void push_dependancy(ILayer<T>* pDependancyLayer);

        virtual ml::Mat<T> getInputFromDependancyOutputs(ml::Mat<T> _in);

        // protected:
        std::string getName() const { return mName; }
        void setName(const std::string& name) { mName = name; }


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

        virtual ml::Mat<T> getOutputByID(ILayer<T>* pID);
        virtual void setOutputByID(ILayer<T>* pID, ml::Mat<T> output);

        virtual int getEpoch() const { return epoch; }
        virtual void setEpoch(int i) { epoch = i; }


    protected:
        size_t mNumInputNodes;  // the num inputs into this layer
        size_t mNumOutputNodes; // the num outputs out of this layer (usually just mInputLayerSize + #biases)

        // weights to each sibling, weights belong to te edge.
        // if NULL then do not apply weights, just forward the input // amps sibling to weights
        std::map<ILayer<T>*, ml::Mat<T>> weights;

        void AddToLayersMap(ILayer<T>* layer);
        // gaurentees destruction uniqueness.. so only deletes once
        // keeps track of all layers ever. So that we can free them all
        // and yet still be able to reuse Layers.. in the network..
        static std::map<ILayer<T>*, ILayer<T>*> mLayersMap;

    public:
        Siblings& getSiblings() { return siblings; }
        // each edge is a directed connection to a sibling.. (think of as next ptrs)
        std::vector<ILayer<T>*> siblings;
        std::vector<ILayer<T>*> dependancies;
        int epoch; // which clock cycle are we in the feeding process

    protected:
        /// (0) We get input fed to us + input from dependancies
        // inputs

        /// (1) we can hold a temporary sum of the inputs in the mActivated var
        // These can be got from each ILayer to weigh as input into sibling ILayers
        ml::Mat<T> mActivated;

        /// (2) then we have weighted activations from the current ILayer
        /// to each of the ILayer's siblings. These are fed into said sibling ILayers
        std::map<ILayer<T>*, ml::Mat<T>> mOutputs; // ouputs to sibling from sibling id

    private:
        std::string mName;
    };

    template <typename T>
    ml::Mat<T> ILayer<T>::getOutputByID(ILayer<T>* pID) {
        auto it = this->mOutputs.find(pID);
        return it != this->mOutputs.end() ? it->second : ml::Mat<T>();
    }

    template <typename T>
    void ILayer<T>::setOutputByID(ILayer<T>* pID, ml::Mat<T> output) {
        if (!pID)
            assert(false);
        this->mOutputs[pID] = output;
    }

    template <typename T>
    std::map<ILayer<T>*, ILayer<T>*> ILayer<T>::mLayersMap = std::map<ILayer<T>*, ILayer<T>*>();


    template <typename T>
    void ILayer<T>::init() {
        this->SetVisited(true);
        ILayer<T>* pSib = NULL;

        for (auto sibIt : this->siblings) {
            pSib = dynamic_cast<ILayer<T>*>(sibIt);
            cout << getName() << ": " << "initializing the weights to the sibling=" << pSib->getName() << endl;
            if (pSib)
                this->initWeights(pSib);
        }
        for (auto sib : this->siblings) {
            pSib = dynamic_cast<ILayer<T>*>(sib);

            if (!pSib->IsVisited()) {
                cout << getName() << ": " << "visiting sibling:" << pSib->getName() << endl;
                pSib->init();
            } else {
                cout << getName() << ": " << "already visited sibling:" << pSib->getName() << endl;
            }
        }
    }


    template <typename T>
    void ILayer<T>::initWeights(ILayer<T>* pSib) {
        const int numInputPerNode = this->getOutputSize();  // num input to each node in next layer
        const int numNodesNextLayer = pSib->getInputSize();   // num nodes in next layer
        const T mean = 0.1, stddev = 0.01;
        // Note: the weights are initialized such that each row is the weight cooeficients for the input into the sibling
        // the sibling has "numNodesNextLayer" therefore, it is clear that there are that many weights. One per next node.
        this->weights[pSib] = ml::initWeightsNormalDist<T>(numNodesNextLayer, numInputPerNode, mean, stddev);
    }


    /* Shoudl get called upon construction of an ILayer.. */
    template <typename T>
    void ILayer<T>::AddToLayersMap(ILayer<T>* layer) {
        this->mLayersMap[layer] = layer;
    }

    template <typename T>
    void ILayer<T>::connect(ILayer<T>* nextLayer) {
        this->siblings.push_back(nextLayer);
        nextLayer->push_dependancy(this);
        cout << this->getName() << " -> " << nextLayer->getName() << endl;
    }

    template <typename T>
    void ILayer<T>::connect(std::vector<ILayer<T>*> nextLayers) {
        const int numLayers = (int) nextLayers.size();
        for (int i = 0; i < numLayers; ++i) {
            this->siblings.push_back(nextLayers[i]);
            nextLayers[i]->push_dependancy(this);
        }
    }


    template <typename T>
    void ILayer<T>::push_dependancy(ILayer<T>* pDependancyLayer) {
        dependancies.push_back(pDependancyLayer);
    }

    /* Given the inputs from this' dependancies and an optional input vector.
    Returns the Mat that contains one row which is the sum of all inputs given.
    TODO: Perhaps the operation can be different. Sum is ok since that is what
    W1*in1 is anyways.  adding what what was output previously together is
    a similar operation. This model assumes that all inputs have the same size (size.cx)
    */
    template <typename T>
    ml::Mat<T> ILayer<T>::getInputFromDependancyOutputs(ml::Mat<T> _in) {
        std::vector<ml::Mat<T>> inputs;
        for (auto dep : this->dependancies) {
            inputs.push_back(dep->getOutputByID(this));
        }
        if (_in.size() != Size(0, 0)) {
            inputs.push_back(_in);
        }
        ml::Mat<T> input = ml::SumRows<T>(inputs);
        return input;
    }


    /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    */

    template <typename T>
    class Layer : public ILayer<T> {
    public:
        typedef ILayer<T> baseclass;
    public:
        Layer(int numNodes, std::string name = "");
        virtual ~Layer();

    protected:
        void common_construct();

        // ILayer<T> overrides
    public:
        virtual void init() override;
        virtual void feed(ml::Mat<T> in, int epoch) override;
        virtual typename ml::Mat<T>::Row activate(typename ml::Mat<T>::Row in) override;
        virtual void activate(int nodeIdx, T in) override;
        virtual ml::Mat<T> getActivatedInput() override;
        virtual void setActivatedInput(ml::Mat<T> activatedInput) override;
    };


    template <typename T>
    Layer<T>::Layer(int numNodes, std::string name) : baseclass() {
        this->setNumInputNodes(numNodes);
        common_construct();
        baseclass::setName(name);
    }

    template <typename T>
    void Layer<T>::common_construct() {
        this->AddToLayersMap(this);
    }

    template <typename T>
    Layer<T>::~Layer() {

    }

    template <typename T>
    void Layer<T>::init() {
        baseclass::init();
    }

    // feeding is weighing and activating across ALL edges from this sib to it's sib
    template <typename T>
    void Layer<T>::feed(ml::Mat<T> _in, int epoch) {
        this->SetVisited(true);
        this->setEpoch(epoch);

        // TODO: update this so it works correctly
        ml::Mat<T> inputMat = this->getInputFromDependancyOutputs(_in);
        if (!inputMat.IsGood())
            return;

        // Add bias units:
		for (int i = 0; i < GetNumBiasNodes(); ++i)
			pushBiasCol<T>(inputMat);

        // perform weight and activation
        this->mActivated = inputMat.Copy(); // todo: activate these with sigmoid or gaussian
        for (auto sib : this->siblings) {
            auto weightIt = this->weights.find(sib);
			// weight the inputMat from this layer to the sibling layer
            if (weightIt != this->weights.end()) {
                // expect row vec (or num cols == num nodes in sibling layer), true for "Is param 2 transposed already"
                this->mOutputs[sib] = ml::Mult<T>(inputMat, weightIt->second, true);
                assert(sib->getInputSize() == this->mOutputs[sib].size().cx);
            }
        }
    }

    template <typename T>
    typename ml::Mat<T>::Row Layer<T>::activate(typename ml::Mat<T>::Row in) {
        return typename ml::Mat<T>::Row();
    }
    template <typename T>
    void Layer<T>::activate(int nodeIdx, T in) {

    }


    template <typename T>
    ml::Mat<T> Layer<T>::getActivatedInput() {
        return this->mActivated;
    }

    template <typename T>
    void Layer<T>::setActivatedInput(ml::Mat<T> activatedInput) {
        this->mActivated = activatedInput;
    }




    /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    */

    template <typename T>
    class Network : public ILayer<T> {
    public:
        typedef ILayer<T> baseclass;
    public:
        // default constructor, use connect(l1, l2) to connect layers
        Network();

        // construct network with layers in order, assumes that
        // each is connected to the next until the output layer
        Network(std::vector<ILayer<T>*> layers);

        virtual ~Network();

    public:
        virtual void setInputLayer(ILayer<T>* pInLayer);
        virtual void setOutputLayer(ILayer<T>* pOutLayer);
        virtual ILayer<T>* getInputLayer() const { return pInputLayer; }
        virtual ILayer<T>* getOutputLayer() const { return pOutputLayer; }


    public:
        virtual void        train(const ml::Mat<T>& samples);
        virtual ml::Mat<T>  feed(const ml::Mat<T>& in);
        virtual void        backprop(const ml::Mat<T>& output_errors);
        virtual ml::Mat<T>  getOutput();
        virtual void        connect(ILayer<T>* l1, ILayer<T>* l2);

        // ILayer<T> overrides
    public:
        virtual void init() override;
        virtual void feed(ml::Mat<T> in, int epoch) override;
        virtual typename ml::Mat<T>::Row activate(typename ml::Mat<T>::Row in) override;
        virtual void activate(int nodeIdx, T in) override;
        virtual ml::Mat<T> getActivatedInput() override;
        virtual void setActivatedInput(ml::Mat<T> activatedInput) override;
        // Network implementation: can connect ILayers together.. which includes {Layer, Network}
        // So, a network can connect multiple networks together into one larger network, recursivly

		/// sizing properties.. overridden to support asking network for the size of it's input and output.
	public:
		virtual void setNumInputNodes(size_t nInputNodes) override;
		virtual size_t getNumInputNodes() const override;
		virtual void setNumOutputNodes(size_t nOutputNodes) override;
		virtual size_t getNumOutputNodes() const override;
		virtual size_t getOutputSize() const override;
		virtual size_t getInputSize() const override;

    protected:
        void common_construct();

    protected:
        void reset_isvisited();

    private:
        ILayer<T>* pInputLayer;
        ILayer<T>* pOutputLayer;
    };



    template <typename T>
    Network<T>::Network() {
        common_construct();
    }

    template <typename T>
    Network<T>::Network(std::vector<ILayer<T>*> layers) {
        common_construct();
    }

    template <typename T>
    void Network<T>::common_construct() {
        ILayer<T>::AddToLayersMap(this);
        pInputLayer = NULL;
        pOutputLayer = NULL;
    }

    template <typename T>
    Network<T>::~Network() {

    }


    template <typename T>
    void Network<T>::setInputLayer(ILayer<T>* pInLayer) {
        pInputLayer = pInLayer;
        cout << pInputLayer->getName() << endl;
    }

    template <typename T>
    void Network<T>::setOutputLayer(ILayer<T>* pOutLayer) {
        pOutputLayer = pOutLayer;
    }


    template <typename T>
    void Network<T>::init() {
        reset_isvisited();
        this->SetVisited(true);
        assert(pInputLayer);
        cout << ILayer<T>::getName() << ": " << "initializing the inner layers' connections" << endl;
        pInputLayer->init();
        cout << ILayer<T>::getName() << ": " << "initializing the intra connections" << endl;
        baseclass::init();
    }

    // call before init and before we need the state to be reset
    template <typename T>
    void Network<T>::reset_isvisited() {
        cout << ILayer<T>::getName() << ": " << "reseting is visited for all INodes" << endl;
        for (auto it : ILayer<T>::mLayersMap) {
            if (it.second) {
                it.second->SetVisited(false);
                it.second->SetAboutToBeVisited(false);
            }
        }
    }


    template <typename T>
    std::stack<ILayer<T>*> makeSiblingStack(const typename ml::ILayer<T>::Siblings& _siblings) {
        std::stack<ILayer<T>*> sibStack;
        auto it = _siblings.begin();
        for (; it != _siblings.end(); ++it)
            sibStack.push(*it);
        return sibStack;
    }

    template <typename T>
    std::stack<ILayer<T>*> makeSiblingStackNotVisited(const typename ml::ILayer<T>::Siblings& _siblings) {
        std::stack<ILayer<T>*> sibStack;
        auto it = _siblings.begin();
        for (; it != _siblings.end(); ++it) {
            ILayer<T>* pSib = *it;
            if (pSib && !pSib->IsVisited() && !pSib->IsAboutToBeVisited()) {
                sibStack.push(pSib);
                // kludge to prevent feeding multiple times the same sibling..
                // if we push it then we have got it ready on the next epoch
                pSib->SetAboutToBeVisited(true);
            }
        }
        return sibStack;
    }

    template <typename T>
    void Network<T>::train(const ml::Mat<T>& samples) {
        for (int i = 0; i < samples.size().cy; ++i) {
            ml::Mat<T> row = ml::Mat<T>(samples.row(i), samples.size().cx);
            feed(row);
            ml::Mat<T> errors;
            backprop(errors);
        }
    }

    template <typename T>
    ml::Mat<T> Network<T>::feed(const ml::Mat<T>& in)  {
        reset_isvisited();
        this->feed(in, 0);
        return this->getOutputByID(this);
    }

    template <typename T>
    void Network<T>::backprop(const ml::Mat<T>& output_errors) {
        /// TODO:..
    }

    template <typename T>
    void Network<T>::feed(ml::Mat<T> in, int epoch) {
        assert(pInputLayer && pOutputLayer);
        this->SetVisited(true);
        this->setEpoch(epoch);

        ml::Mat<T> input = this->getInputFromDependancyOutputs(in);
            // network has siblings, other networks or layers, then layers have siblings (other layers)

        // feed the input layer to initialize the output
        pInputLayer->feed(input, epoch);

        std::stack<ILayer<T>*> nextStack;
        std::stack<ILayer<T>*> sibStack = makeSiblingStackNotVisited<T>(pInputLayer->siblings);

        // get ready for the next epoch of nodes to feed
        epoch++;

        while (!sibStack.empty()) {
            ILayer<T>* pSib = sibStack.top();
            sibStack.pop();
            nextStack = joinStacks<ILayer<T>*>(nextStack,
                makeSiblingStackNotVisited<T>(pSib->siblings));
            pSib->feed(ml::Mat<T>(), epoch);
            if (sibStack.empty()) {
                sibStack = nextStack;
                emptyOut<ILayer<T>*>(nextStack);
                epoch += 1;
            }
        }
    }

    template <typename T>
    typename ml::Mat<T>::Row Network<T>::activate(typename ml::Mat<T>::Row in) {
        return typename ml::Mat<T>::Row();
    }

    template <typename T>
    void Network<T>::activate(int nodeIdx, T in) {

    }

    template <typename T>
    ml::Mat<T> Network<T>::getActivatedInput() {
        if (!pOutputLayer) return ml::Mat<T>();
        return pOutputLayer->getActivatedInput();
    }

    template <typename T>
    void Network<T>::setActivatedInput(ml::Mat<T> activatedInput) {
        if (!pOutputLayer) return;
        pOutputLayer->setActivatedInput(activatedInput);
    }


	/// Sizing 
    template <typename T>
	void Network<T>::setNumInputNodes(size_t nInputNodes)
	{
		ILayer<T>* pLayer = getInputLayer();
		if (pLayer)
			pLayer->setNumInputNodes(nInputNodes);
	}

    template <typename T>
	size_t Network<T>::getNumInputNodes() const 
	{ 
		ILayer<T>* pLayer = getInputLayer();
		return pLayer ? pLayer->getNumInputNodes() : 0;
	}

    template <typename T>
	void Network<T>::setNumOutputNodes(size_t nOutputNodes)
	{
		ILayer<T>* pLayer = getOutputLayer();
		if (pLayer)
			pLayer->setNumOutputNodes(nOutputNodes);
	}

    template <typename T>
	size_t Network<T>::getNumOutputNodes() const 
	{ 
		ILayer<T>* pLayer = getOutputLayer();
		return pLayer ? pLayer->getNumOutputNodes() : 0;
	}

    template <typename T>
	size_t Network<T>::getOutputSize() const 
	{ 
		return getNumOutputNodes(); 
	}

    template <typename T>
	size_t Network<T>::getInputSize() const 
	{ 
		return getNumInputNodes(); 
	}

	/// End sizing

	
    template <typename T>
    ml::Mat<T> Network<T>::getOutput() {
        return getActivatedInput();
    }

    template <typename T>
    void Network<T>::connect(ILayer<T>* l1, ILayer<T>* l2) {
        assert(l1 && l2);
        l1->connect(l2);
    }

} // namespace ml
