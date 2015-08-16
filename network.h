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


    template <typename T>
    ml::Mat<T> Sigmoid(ml::Mat<T> mat) {
        return mat.Copy(); // TODO: implement as a sigmoid
    }

    template <typename T>
    ml::Mat<T> SigGrad(ml::Mat<T> mat) {
        return ml::ElementMult(mat, ml::Diff<T>(1, mat)); 
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
        typedef std::vector<ILayer<T>*> LayerVector;

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
        virtual ml::Mat<T> getInput() = 0;
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
        static size_t   GetNumBiasNodes() { return 1; }
        virtual void    setNumInputNodes(size_t nInputNodes)
        {
            mNumInputNodes = nInputNodes;
            mNumOutputNodes = (mNumInputNodes + GetNumBiasNodes());
        }
        virtual size_t  getNumInputNodes() const { return mNumInputNodes; }
        virtual void    setNumOutputNodes(size_t nOutputNodes)
        {
            mNumOutputNodes = nOutputNodes;
            mNumInputNodes = (mNumOutputNodes - GetNumBiasNodes());
        }
        virtual size_t getNumOutputNodes() const { return mNumOutputNodes; }
        virtual size_t getOutputSize() const { return getNumOutputNodes(); }
        virtual size_t getInputSize() const { return getNumInputNodes(); }

        virtual ml::Mat<T> getOutputByID(ILayer<T>* pID);
        virtual void setOutputByID(ILayer<T>* pID, ml::Mat<T> output);

        virtual int  getEpoch() const { return epoch; }
        virtual void setEpoch(int i) { epoch = i; }

        // returns the weights that weigh the output from this layer to the next layer
        virtual ml::Mat<T> getWeights(ILayer<T>* pNextLayer);

    protected:
        size_t mNumInputNodes;  // the num inputs into this layer
        size_t mNumOutputNodes; // the num outputs out of this layer (usually just mInputLayerSize + #biases)

        // weights to each sibling, weights belong to te edge.
        // if NULL then do not apply weights, just forward the input // amps sibling to weights
        std::map<ILayer<T>*, ml::Mat<T>> weights;

        void AddToLayersMap(ILayer<T>* layer);
        void resetAllIsVisited();
        /// Keep track of all layers ever connected to a network.. If it is in here then it 
        /// is owned by some network.  If it is not here then it was never added to a network. 
        /// Layers in here will be deleted..
        static std::map<ILayer<T>*, ILayer<T>*> mLayersMap;

    public:
        void setErrors(ml::Mat<T> errors) { m_errors = errors; }
        ml::Mat<T> getErrors() const { return m_errors; }
    protected:
        ml::Mat<T> m_errors;

    public:
        LayerVector& getSiblings() { return siblings; }
        // each edge is a directed connection to a sibling.. (think of as next ptrs)
        std::vector<ILayer<T>*> siblings;
        std::vector<ILayer<T>*> dependancies;
        int epoch; // which clock cycle are we in the feeding process

    protected:
        /// (0) We get input fed to us + input from dependancies
        ml::Mat<T> mInput;

        /// (1) the activated inputs into this layer. Will be weighted and sent as 
        /// output to next layers. Could be considered "output" from a network
        ml::Mat<T> mActivated;

        /// (2) then we have weighted activations from the current ILayer
        /// to each of the ILayer's siblings. These are fed into said sibling ILayers
        std::map<ILayer<T>*, ml::Mat<T>> mOutputs; // ouputs to sibling from sibling id

    private:
        std::string mName;
    };

    template <typename T>
    void ILayer<T>::resetAllIsVisited() {
        cout << getName() << ": " << "reseting is visited for all INodes" << endl;
        for (auto it : mLayersMap) {
            if (it.second) {
                it.second->SetVisited(false);
                it.second->SetAboutToBeVisited(false);
            }
        }
    }

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

    /// Static Map
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

    template <typename T>
    ml::Mat<T> ILayer<T>::getWeights(ILayer<T>* pNextLayer) {
        auto it = weights.find(pNextLayer);
        if(it != weights.end())
            return it->second;
        return ml::Mat<T>();
    }

    /* Should get called upon construction of an ILayer.. */
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
        virtual ml::Mat<T> getInput() override;
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
        //this->AddToLayersMap(this);
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

        // Activate the input 
        /// todo: activate these with sigmoid or gaussian
        //setActivatedInput(inputMat.Copy());
        this->mInput     = inputMat.Copy();
        this->mActivated = Sigmoid<T>(this->mActivated);

        // Add bias units:
        for (int i = 0; i < this->GetNumBiasNodes(); ++i)
            pushBiasCol<T>(inputMat);

        // weight the activated node values and provide weighted sums to next layer's nodes
        for (auto sib : this->siblings) {
            auto weightIt = this->weights.find(sib);
            if (weightIt != this->weights.end()) {
                this->setOutputByID(sib, ml::Mult<T>(inputMat, weightIt->second, true));
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
    ml::Mat<T> Layer<T>::getInput() {
        return this->mInput; 
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
        virtual void        setInputLayer(ILayer<T>* pInLayer);
        virtual void        setOutputLayer(ILayer<T>* pOutLayer);
        virtual ILayer<T>*  getInputLayer() const { return pInputLayer; }
        virtual ILayer<T>*  getOutputLayer() const { return pOutputLayer; }


    public:
        virtual void        train(const ml::Mat<T>& samples, const ml::Mat<T>& nominals);
        virtual ml::Mat<T>  feed(const ml::Mat<T>& in);
        virtual void        backprop();
        virtual ml::Mat<T>  getOutput();
        virtual void        connect(ILayer<T>* l1, ILayer<T>* l2);
        virtual void        connect(ILayer<T>* nextLayer);

        // ILayer<T> overrides
    public:
        virtual void        init() override;
        virtual void        feed(ml::Mat<T> in, int epoch) override;
        virtual typename ml::Mat<T>::Row activate(typename ml::Mat<T>::Row in) override;
        virtual void        activate(int nodeIdx, T in) override;
        virtual ml::Mat<T>  getInput() override;
        virtual ml::Mat<T>  getActivatedInput() override;
        virtual void        setActivatedInput(ml::Mat<T> activatedInput) override;
        // Network implementation: can connect ILayers together.. which includes {Layer, Network}
        // So, a network can connect multiple networks together into one larger network, recursivly

		/// sizing properties.. overridden to support asking network for the size of it's input and output.
	public:
		virtual void      setNumInputNodes(size_t nInputNodes) override;
		virtual size_t    getNumInputNodes() const override;
		virtual void      setNumOutputNodes(size_t nOutputNodes) override;
		virtual size_t    getNumOutputNodes() const override;
		virtual size_t    getOutputSize() const override;
		virtual size_t    getInputSize() const override;

    public:
        virtual ml::Mat<T> getOutputByID(ILayer<T>* pID);

    protected:
        void common_construct();

    protected:
        void addOutputLayerSiblings(ILayer<T>* nextLayer);

    protected:
        void updateLayersMaps(ILayer<T>* layer);        // updates both global map and instance map
        void AddToNetworkLayersMap(ILayer<T>* layer);   // adds to instance map ref to layer to own
        void resetNetworkIsVisited();                   // resets this' own layers' visited flags..
        std::map<ILayer<T>*, ILayer<T>*> mNetworkLayersMap; // local to instance, vs global to all layers see: mLayersMap

    private:
        std::vector<ILayer<T>*> mOutputLayerSiblingCache;

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
        //ILayer<T>::AddToLayersMap(this);
        pInputLayer = NULL;
        pOutputLayer = NULL;
    }

    template <typename T>
    Network<T>::~Network() {
        mOutputLayerSiblingCache.clear();
    }


    template <typename T>
    void Network<T>::setInputLayer(ILayer<T>* pInLayer) {
        pInputLayer = pInLayer;
        cout << pInputLayer->getName() << endl;
    }

    template <typename T>
    void Network<T>::setOutputLayer(ILayer<T>* pOutLayer) {
        pOutputLayer = pOutLayer;
        if (mOutputLayerSiblingCache.size() > 0 && pOutputLayer != NULL) {
            for (ILayer<T>* pLayer : mOutputLayerSiblingCache)
                pOutputLayer->connect(pLayer);
            mOutputLayerSiblingCache.clear();
        }
    }


    template <typename T>
    void Network<T>::init() {
        resetNetworkIsVisited();
        this->SetVisited(true);
        assert(pInputLayer);
        cout << ILayer<T>::getName() << ": " << "initializing the inner layers' connections" << endl;
        pInputLayer->init();
        cout << ILayer<T>::getName() << ": " << "initializing the intra connections" << endl;
        baseclass::init();
    }

    // call before init and before we need the state to be reset
    template <typename T>
    void Network<T>::resetNetworkIsVisited() {
        cout << ILayer<T>::getName() << ": " << "reseting is visited for this network's INodes" << endl;
        for (auto it : this->mNetworkLayersMap) {
            if (it.second) {
                it.second->SetVisited(false);
                it.second->SetAboutToBeVisited(false);
            }
        }
    }

    template <typename T>
    void Network<T>::AddToNetworkLayersMap(ILayer<T>* layer) {
        this->mNetworkLayersMap[layer] = layer;
    }

    template <typename T>
    std::stack<ILayer<T>*> makeLayerStack(const typename ml::ILayer<T>::LayerVector& layers) {
        std::stack<ILayer<T>*> sibStack;
        auto it = layers.begin();
        for (; it != layers.end(); ++it)
            sibStack.push(*it);
        return sibStack;
    }

    template <typename T>
    std::stack<ILayer<T>*> makeLayerStackWithUnvisited(const typename ml::ILayer<T>::LayerVector& layers) {
        std::stack<ILayer<T>*> sibStack;
        auto it = layers.begin();
        for (; it != layers.end(); ++it) {
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

    template <class ItemType>
    ItemType getNextItem(std::stack<ItemType> s) {
        ItemType item = s.top(); 
        s.pop();
        return item;
    }


    template <typename T>
    void Network<T>::train(const ml::Mat<T>& samples, const ml::Mat<T>& nominals) {
        ILayer<T>::resetAllIsVisited();
        for (int i = 0; i < samples.size().cy; ++i) {
            ml::Mat<T> row = ml::Mat<T>(samples.row(i), samples.size().cx);
            ml::Mat<T> predicted = feed(row);
            getOutputLayer()->setErrors(ml::Diff(nominals, predicted));
            backprop();
        }
    }

    template <typename T>
    ml::Mat<T> Network<T>::feed(const ml::Mat<T>& in)  {
        ILayer<T>::resetAllIsVisited();
        this->feed(in, 0);
        return this->getOutput();
    }

    template <typename T>
    void Network<T>::backprop() {
        /* 
        in a network:
        for each dependancy:
            propogate the errors to the depdenancy by weighting this layer's error
        */
        typedef std::stack<ILayer<T>*> Stack;
        Stack toVisit;
        toVisit.push(getOutputLayer());
        do {
            // propogate errors backwards from cur layer to it's dependancies
            ILayer<T>* pCurLayer = getNextItem(toVisit);
            ml::Mat<T> errors = pCurLayer->getErrors();
            for(ILayer<T>* pPrevLayer : pCurLayer->dependancies) {
                ml::Mat<T> weights = pPrevLayer->getWeights(pCurLayer);
                weights.Transpose();
                ml::Mat<T> deltaSig = SigGrad<T>(pPrevLayer->getInput());
                ml::Mat<T> weightedErr = weights.Mult(errors);
                ml::Mat<T> gradientErr = ml::ElementMult<T>(weightedErr, deltaSig);
                pPrevLayer->setErrors(gradientErr);
            }

            // keep track of the layers to visit. also update current layer pointer
            Stack unvisited = makeLayerStackWithUnvisited<T>(pCurLayer->dependancies); 
            toVisit = joinStacks<ILayer<T>*>(toVisit, unvisited);
            pCurLayer = getNextItem(toVisit);
        } while(!toVisit.empty());
    }

    template <typename T>
    void Network<T>::feed(ml::Mat<T> in, int epoch) {
        resetNetworkIsVisited();
        assert(pInputLayer && pOutputLayer);
        this->SetVisited(true);
        this->setEpoch(epoch);

        ml::Mat<T> input = this->getInputFromDependancyOutputs(in);
            // network has siblings, other networks or layers, then layers have siblings (other layers)

        // feed the input layer to initialize the output
        pInputLayer->feed(input, epoch);

        std::stack<ILayer<T>*> nextStack;
        std::stack<ILayer<T>*> sibStack = makeLayerStackWithUnvisited<T>(pInputLayer->siblings);

        // get ready for the next epoch of nodes to feed
        epoch++;

        while (!sibStack.empty()) {
            ILayer<T>* pSib = sibStack.top();
            sibStack.pop();
            nextStack = joinStacks<ILayer<T>*>(nextStack,
                makeLayerStackWithUnvisited<T>(pSib->siblings));
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
    ml::Mat<T> Network<T>::getInput() {
        if (!pInputLayer) return ml::Mat<T>();
        return pInputLayer->getInput();
    }

    template <typename T>
    ml::Mat<T> Network<T>::getActivatedInput() {
        if (!pInputLayer) return ml::Mat<T>();
        return pInputLayer->getActivatedInput();
    }

    template <typename T>
    void Network<T>::setActivatedInput(ml::Mat<T> activatedInput) {
        if (!pInputLayer) return;
        pInputLayer->setActivatedInput(activatedInput);
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
    ml::Mat<T> Network<T>::getOutputByID(ILayer<T>* pID) {
        ILayer<T>* pLayer = getOutputLayer();
        return pLayer ? pLayer->getOutputByID(pLayer) : ml::Mat<T>();
    }

	
    template <typename T>
    ml::Mat<T> Network<T>::getOutput() {
        if (!pOutputLayer) return ml::Mat<T>();
        return pOutputLayer->getActivatedInput();
    }

    template <typename T>
    void Network<T>::updateLayersMaps(ILayer<T>* layer) {
        // keep track of the layers added to this network, if added to this before another 
        // network then the other network will not have the layer in it's network layers map.. 
        // however the mLayersMap will still have all layers. If the layers map has it then a 
        // network has already claimed the layer as its own.
        if (ILayer<T>::mLayersMap.find(layer) == ILayer<T>::mLayersMap.end()) {
            if (this->mNetworkLayersMap.find(layer) == this->mNetworkLayersMap.end()) {
                AddToNetworkLayersMap(layer);
            }
            ILayer<T>::AddToLayersMap(layer);
        }
    }

    template <typename T>
    void Network<T>::connect(ILayer<T>* l1, ILayer<T>* l2) {
        assert(l1 && l2);
        l1->connect(l2);
        updateLayersMaps(l1);
        updateLayersMaps(l2);
    }

    template <typename T>
    void Network<T>::connect(ILayer<T>* nextLayer) {
        ILayer<T>* pLayer = this->getOutputLayer();
        if (pLayer) {   // connect the output to the sibling right away if possible
            pLayer->connect(nextLayer);
        } else {        // otherwise cache it for later, when we set output then we check the sibling cache
            this->addOutputLayerSiblings(nextLayer);
        }
    }

    template <typename T>
    void Network<T>::addOutputLayerSiblings(ILayer<T>* sibling) {
        mOutputLayerSiblingCache.push_back(sibling);
    }

} // namespace ml
