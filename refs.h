#pragma once

/*
  template <typename T>
  class refArr ..

  refArr<int> referenceCountedArray(new int[12321], 12321);
  {
  refArr<int> data = referenceCountedArray;
  doStuff(data);
  }
  */

namespace ml {
	template <typename T>
	class sharedRef {
	public:
		sharedRef(T* v, int count = 1) {
			vals = v;
			refs = count;
		}
		T* vals;
		int refs;
	};

	template <typename T>
	class refArr {
	public:
		refArr(){
			mShared = NULL;
			mVals = NULL;
			mSize = 0;
		}
		refArr(T* v, int size) {
			mShared = new sharedRef<T>(v);
			mVals = v; // just for quick reference
			mSize = size;
		}
		refArr(const refArr<T>& ref) {
			if (!!ref.mVals || !!ref.mShared->vals) {
				mShared = ref.mShared;
				increment_refcount();
				mVals = ref.mShared->vals;
				mSize = ref.mSize;
			}
		}
		virtual ~refArr() {
			if (mVals) {
				decrement_refcount();
				destroy_ifnoref();
				reset_vals(NULL);
			}
		}

		void setSize(size_t size) {
			mSize = size;
		}
		size_t getSize() const {
			return mSize;
		}

		bool operator==(T* v) const {
			return mVals == v;
		}
		bool operator!=(T* v) const {
			return mVals != v;
		}
		bool operator==(const refArr<T>& a1) const {
			return mShared == a1.mShared;
		}
		bool operator!=(const refArr<T>& a1) const {
			return mShared != a1.mShared;
		}
		refArr<T>& operator=(refArr<T>& a1) {
			if (a1.mShared == this->mShared)
				return *this;
			this->decrement_refcount();
			this->destroy_ifnoref();
			this->reset_vals(NULL);
			if (a1.mShared == NULL)
				return *this;
			reset_shared(a1.get_shared());
			reset_vals(this->mShared->vals);
			increment_refcount();
			return *this;
		}
		// should call set size after this..
		refArr<T>& operator=(T* v) {
			this->decrement_refcount();
			this->destroy_ifnoref();
			this->reset_vals(NULL);
			this->mVals = v;
			if (!v)
				return *this;
			this->reset_shared(new sharedRef<T>(v));
            return *this;
		}
		T& operator[](const int idx) const {
			return mVals[idx];
		}
		inline T* data() const {
			return mVals;
		}
		T* copy() const {
			T* c = new T[mSize];
			memcpy(c, mVals, sizeof(T) * mSize);
			return c;
		}

		// If last ref then does not delete,
		// just delete sharedRef and reutrn the values
		T* release() {
			if (!mVals)
				return NULL;
			decrement_refcount();
			destroy_shared_ifnorefs();
			T* v = mVals;
			reset_vals(NULL);
			return v;
		}

		int numReferences() const {
			if (!mShared)
				return 0;
			return mShared->refs;
		}
		/// -- -- -- -- -- ///

		void print() const {
			if (!mVals) return;
			using namespace std;
			for (size_t i = 0; i < mSize; i++) {
				cout << mVals[i] << " ";
			} cout << endl;
		}

		/// -- -- -- -- -- ///

	protected:
		inline void decrement_refcount() {
			if (!mShared)
				return;
			--mShared->refs;
		}

		void destroy_ifnoref() {
			if (mShared && mShared->refs == 0
				&& mShared->vals) {
				delete[] mShared->vals;
				mShared->vals = NULL;
			}
			destroy_shared_ifnorefs();
			// values are deleted too so might as well set to NULL
			reset_vals(NULL);
		}

		inline void destroy_shared_ifnorefs() {
			if (!mShared)
				return;
			if (mShared->refs == 0)
				delete mShared;
			reset_shared(NULL);
		}

		inline void reset_vals(T* v) {
			mVals = v;
		}

		inline void reset_shared(sharedRef<T>* shared) {
			mShared = shared;
		}

		inline sharedRef<T>* get_shared() const {
			return mShared;
		}

		inline void increment_refcount() {
			if (!mShared)
				return;
			++mShared->refs;
		}

	protected:
		T* mVals;
		size_t mSize;
		sharedRef<T>* mShared;
	};
}
