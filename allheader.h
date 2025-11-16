#pragma once

// C
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

// C++
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <initializer_list>

// dependancies
// /Arch
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

// openmp
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

// client includes
#include "utility.h"
#include "refs.h"
#include "timer.h"
#include "math/nn_math.h"
#include "math/rect.h"
#include "Matrix/matrix.h"
