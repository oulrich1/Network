#pragma once

#ifndef uint
#define uint unsigned int
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

template <typename Vec>
void copy(Vec& src, Vec& dst) {
  dst.reserve(src.size());
  std::copy(src.begin(), src.end(), std::back_inserter(dst));
}

template <typename T, typename Vec>
T dot(const Vec& v1, const Vec& v2) {
  const size_t vec_size = v1.size();
  T sum = T(0.0);
  for (int i = 0; i < vec_size; ++i)
    sum += v1[i] * v2[i];
  return sum;
}

template <typename T>
T dot(const T* v1, const T* v2, const size_t size) {
  T sum = T(0.0);
  for (size_t i = 0; i < size; ++i)
    sum += v1[i] * v2[i];
  return sum;
}


#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
inline __m128 sse_dot4(__m128 v0, __m128 v1) {
    v0 = _mm_mul_ps(v0, v1);
    v1 = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(2, 3, 0, 1));
    v0 = _mm_add_ps(v0, v1);
    v1 = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(0, 1, 2, 3));
    v0 = _mm_add_ps(v0, v1);
    return v0;
}

#ifdef _WIN32
template <typename Vec>
inline __m128 sse_dot_vec(Vec v1, Vec v2){
	__m128 v = _mm_set1_ps(0.f);
	return v;
}

template <typename Vec>
Vec make_mm_vec(typename Vec::value_type val, size_t size){
	return Vec();
}
#else
template <typename Vec>
inline __m128 sse_dot_vec(Vec v1, Vec v2){
  size_t vec_size = v1.size();
  __m128 sum = _mm_set1_ps(0.f);
  for (uint i = 0; i < vec_size; ++i)
    sum = sum + sse_dot4(v1[i], v2[i]);
  return sum;
}

template <typename Vec>
Vec make_mm_vec(typename Vec::value_type val, size_t size){
  Vec v;
  for (int i = 0; i < size; ++i)
    v.push_back(val);
  return v;
}
#endif
#endif // x86/x64 architecture check

//


template <typename T>
T GCD_naive(T a, T b) {
  // a / v == b / v | v is max
  T l = std::min(a, b);
  for (; l >=0; --l) {
    if(a%l == 0 && b%l == 0) {
      return l;
    }
  }
  return 1;
}

template <typename T>
T GCD(T a, T b) {
  if (a < b) {
    std::swap(a, b);
  }

  T r = a % b;
  T prev_r = a;
  while (r > 0) {
    prev_r = r;
    r = a % b;
    a = b;
    b = r;
  }
  return prev_r;
}

template <typename T>
T LCM(T a, T b) {
  return a*b/GCD<T>(a, b);
}
