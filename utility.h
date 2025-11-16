#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stack>

#ifdef  _DEBUG
#define DEBUG(t, message) 		(Utility::Debug(t, message))
#ifndef DWARNING
#define DWARNING(t, message) 	(Utility::Warning(t, message))
#define DERROR(t, message) 		(Utility::Error(t, message))
#endif
#else
#define DEBUG(t, message) 		((void)0)
#ifndef DWARNING
#define DWARNING(t, message) 	((void)0)
#define DERROR(t, message) 		((void)0)
#endif
#endif

typedef unsigned char uchar;

// returns true if a is within e of b
#define Within(a, b, e) ((((a) <= ((b)+(e))) && ((a) >= ((b)-(e)))) ? 1 : 0)

// enumerate indecies
#define FOR_IDX(_i_, vec) for(unsigned int _i_ = 0; _i_ < vec.size(); ++_i_)

#define CV_PRECISE CV_32FC1
//#define CV_PRECISE CV_64FC1

namespace Utility {

	template <typename T>
	void print_array(T* arr, size_t s) {
		using namespace std;
		cout << "{ ";
		for (size_t i = 0; i < s; i++) {
			cout << arr[i] << " ";
			if(i < s-1) {
				cout << ",";
			}
		}
		cout << "}";
		cout << endl;
	}

	inline void print(const char* str) {
		printf("%s\n", str);
	}

	/// function forward declarations
	inline int Debug(const char* t, const char* message);
	inline int Warning(const char* t, const char* message);
	inline int Error(const char* t, const char* message);

	inline int Debug(const char* t, const char* message) {
		if (!t || !message)
			return -1;
		printf(">> Debug: %s:: %s\n", t, message);
		return 0; // return debug_level;
	}

	inline int Warning(const char* t, const char* message) {
		if (!t || !message)
			return -1;
		printf(">> Warning: %s:: %s\n", t, message);
		return 1; // return debug_level;
	}

	inline int Error(const char* t, const char* message) {
		if (!t || !message)
			return -1;
		printf(">> Error: %s:: %s\n", t, message);
		return 2; // return debug_level;
	}

} // utility


//
//  Open MP Helpers
#if defined(_OPENMP)
int get_num_threads() {
	int nthreads = 0;
	int tid = 0;
#pragma omp parallel private(tid)
	{
		tid = omp_get_thread_num();
		if (tid == 0) {
			nthreads = omp_get_num_threads();
			//  printf("number of threads: %d\n", nthreads);
		}
	}
	return nthreads;
}
#else
inline int get_num_threads() {
	return 1;
}
#endif
//


namespace StackEx {
    /* returns a reversed s1 stack */
    template <typename T>
    std::stack<T> reverseStack(const std::stack<T>& s) {
        std::stack<T> s1 = s;
        std::stack<T> s2;
        while (!s1.empty()) {
            s2.push(s1.top());
            s1.pop();
        }
        return s2;
    }

    /* Joines the stacks s1 and s2: new stack == {s1, s2} where the right side is the top of a generic stack */
    template <typename T>
    std::stack<T> joinStacks(const std::stack<T>& s1, const std::stack<T>& s2) {
        std::stack<T> joined = s1;
        std::stack<T> s3 = reverseStack(s2);
        while (!s3.empty()) {
            joined.push(s3.top());
            s3.pop();
        }
        return joined;
    }

    template <typename T>
	void emptyOut(std::stack<T>& s) {
		while (!s.empty())
			s.pop();
	}
}


#ifndef BEGIN_TESTS
#define BEGIN_TESTS(title) 								\
{																					\
	printf("\n");														\
	unsigned len = strlen(title);						\
	unsigned bar_size = len + 10;						\
	char* bar = 0;													\
	bar = (char*) malloc(bar_size + 1);			\
	memset(bar, '>', 3);										\
	memset(bar+3, '-', bar_size-6);					\
	memset(&(bar[bar_size-3]), '<', 3);			\
	bar[bar_size] = 0;											\
	printf("%s\n", bar); 										\
	free(bar);															\
	bar = NULL;															\
	bar = (char*) malloc(len + 1);					\
	memset(bar, ' ', len);									\
	bar[len] = 0;														\
	printf(">>>  %s  <<<\n", bar); 					\
	printf(">>>  %s  <<<\n", title);				\
	printf(">>>  %s  <<<\n", bar); 					\
	free(bar);															\
	bar = (char*) malloc(bar_size + 1);			\
	memset(bar, '>', 3);										\
	memset(bar+3, '-', bar_size-6);					\
	memset(&(bar[bar_size-3]), '<', 3);			\
	bar[bar_size] = 0;											\
	printf("%s\n", bar); 										\
	free(bar);															\
}
#else
#warning "BEGIN_TESTS already defined.. Tests Broken"
#endif



#define DEFINE_SIBLING_MAP(CLASS, T)  \
	template <>				\
	std::map<CLASS<T>*, CLASS<T>*> CLASS<T>::mLayersMap = std::map<CLASS<T>*, CLASS<T>*>();




#endif // UTILITY_H
