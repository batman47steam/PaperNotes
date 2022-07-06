#pragma once
#include <stdio.h>
#include <memory.h> // memcpy(), memset()
#include <vector>
#include <string>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include <cmath>
#include <algorithm>

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef long long int64;
typedef unsigned long long uint64;

#define USE_AVX //comment this line to disable __m256; some AVX instructions are not supported by pre-VS2013 compilers.

#ifndef USE_AVX //USE_SSE, based on __m128
//#include <xmmintrin.h> //SSE
//#include <emmintrin.h> //SSE2 //year.2006
//#include <pmmintrin.h> //SSE3
//#include <tmmintrin.h>//SSSE3
#include <smmintrin.h>//SSE4.1 //year.2008
#include <nmmintrin.h>//SSE4.2
#define _align_sz 16 //AVX requires 32bit-alignment
static inline bool _m_aligned(const void *x) { return !(size_t(x) & 15); } //tests whether x is 16byte aligned
static inline bool _m_aligned(size_t x) { return !(x & 15); }
//Define wrappers for float data operations
#define _m_clear      
#define __m             __m128
#define _m_inc          4 //4 floats
static inline int _m_res(int x) { return x & 3; } //x % 4 = x & 3
static inline int _m_nsse(int x) { return x >> 2; } //x / 4 = (x - x&3) >> 2
#define _m_zero         _mm_setzero_ps
#define _m_set1         _mm_set1_ps
#define _m_load         _mm_load_ps
#define _m_loadu        _mm_loadu_ps
#define _m_store        _mm_store_ps
#define _m_storeu       _mm_storeu_ps
#define _m_add          _mm_add_ps
#define _m_sub          _mm_sub_ps
#define _m_mul          _mm_mul_ps
#define _m_div          _mm_div_ps
#define _m_inv          _mm_rcp_ps
#define _m_sqrt         _mm_sqrt_ps
#define _m_rsqrt        _mm_rsqrt_ps
#define _m_min          _mm_min_ps
#define _m_max          _mm_max_ps
#define _m_and          _mm_and_ps
#define _m_or           _mm_or_ps
#define _m_xor          _mm_xor_ps
#define _m_andnot       _mm_andnot_ps
#define _m_eq           _mm_cmpeq_ps
#define _m_neq          _mm_cmpneq_ps
#define _m_lt           _mm_cmplt_ps
#define _m_le           _mm_cmple_ps
#define _m_cvtepi32     _mm_cvtps_epi32 //float -> int, round
#define _m_cvttepi32    _mm_cvttps_epi32 //float -> int, truncation to zero, if out of integer range, will throw an error
//Define wrappers for integer data operations
#define __mi            __m128i
#define _mi_zero        _mm_setzero_si128
#define _mi_set1        _mm_set1_epi32
#define _mi_load        _mm_load_si128
#define _mi_loadu       _mm_loadu_si128
#define _mi_store       _mm_store_si128
#define _mi_storeu      _mm_storeu_si128
#define _mi_add         _mm_add_epi32
#define _mi_sub         _mm_sub_epi32
#define _mi_mul         _mm_mullo_epi32 //mul(a,b) = [a3*b3 a2*b2 a1*b1 a0*b0]
#define _mi_sr          _mm_srli_si128 //shift-right(a,n) shifts "a" right by "n" bits
#define _mi_sr64        _mm_srli_epi64 //shift-right(a,n) shifts every int64 in "a" right by "n" bits
#define _mi_mul64       _mm_mul_epi32 //mul64(a,b) = [high(a2*b2) low(a2*b2) high(a0*b0) low(a0*b0)] = [int64(a2)*int64(b2) int64(a0)*int64(b0)]
#define _mi_mul64p(a,b) _mi_mul64(_mi_sr(a,4),_mi_sr(b,4)) //mul64p(a,b) = [int64(a3)*int64(b3) int64(a1)*int64(b1)]
#define _mi_6432(a,b)   _mm_unpacklo_epi32(_mm_shuffle_epi32(a, shufl), _mm_shuffle_epi32(b,shufl)) //convert 4 int64 (int a&b) to 4 int32
#define _mi_and         _mm_and_si128
#define _mi_or          _mm_or_si128
#define _mi_xor         _mm_xor_si128 //xor(a,b) : a!=b
#define _mi_andnot      _mm_andnot_si128 //andnot(a,b) : (~a) & b
#define _mi_eq          _mm_cmpeq_epi32
#define _mi_gt          _mm_cmpgt_epi32
#define _mi_cvtps       _mm_cvtepi32_ps //int -> float
#define _m_sum(x) (*(x) + *(x + 1) + *(x + 2) + *(x + 3)) //x is float[4]
#define _m_sumd(x) ((double)(*(x)) + *(x + 1) + *(x + 2) + *(x + 3)) //x is float[4]

#else //USE_AVX, based on __m256
#include <immintrin.h>
#define _align_sz 32
static inline bool _m_aligned(const void *x) { return !(size_t(x) & 31); } //return true if x is 32byte aligned
static inline bool _m_aligned(size_t x) { return !(x & 31); }
//Define wrappers for float data operations
#define _m_clear        _mm256_zeroupper() //_mm256_zeroall()
#define __m             __m256
#define _m_inc          8 //8 floats
static inline int _m_res(int x) { return x & 7; } // x % 8 = x & 7
static inline int _m_nsse(int x) { return x >> 3; } //x / 8 = (x - x&7) >> 3
#define _m_zero         _mm256_setzero_ps
#define _m_set1         _mm256_set1_ps
#define _m_load         _mm256_load_ps
#define _m_loadu        _mm256_loadu_ps
#define _m_store        _mm256_store_ps
#define _m_storeu       _mm256_storeu_ps
#define _m_add          _mm256_add_ps
#define _m_sub          _mm256_sub_ps
#define _m_mul          _mm256_mul_ps
#define _m_div          _mm256_div_ps
#define _m_inv          _mm256_rcp_ps
#define _m_sqrt         _mm256_sqrt_ps
#define _m_rsqrt        _mm256_rsqrt_ps
#define _m_min          _mm256_min_ps
#define _m_max          _mm256_max_ps
#define _m_and          _mm256_and_ps
#define _m_or           _mm256_or_ps
#define _m_xor          _mm256_xor_ps
#define _m_andnot       _mm256_andnot_ps
#define _m_eq(a,b)      _mm256_cmp_ps(a, b, _CMP_EQ_OQ)
#define _m_neq(a,b)     _mm256_cmp_ps(a, b, _CMP_NEQ_OQ)
#define _m_lt(a,b)      _mm256_cmp_ps(a, b, _CMP_LT_OQ) 
#define _m_le(a,b)      _mm256_cmp_ps(a, b, _CMP_LE_OQ)
#define _m_cvtepi32     _mm256_cvtps_epi32
#define _m_cvttepi32    _mm256_cvttps_epi32
//Define wrappers for integer data operations
#define __mi            __m256i
#define _mi_zero        _mm256_setzero_si256
#define _mi_set1        _mm256_set1_epi32
#define _mi_load        _mm256_load_si256
#define _mi_loadu       _mm256_loadu_si256
#define _mi_store       _mm256_store_si256
#define _mi_storeu      _mm256_storeu_si256
#define _mi_add         _mm256_add_epi32
#define _mi_sub         _mm256_sub_epi32
#define _mi_mul         _mm256_mullo_epi32
#define _mi_sr          _mm256_srli_si256
#define _mi_sr64        _mm256_srli_epi64
#define _mi_mul64       _mm256_mul_epi32
#define _mi_mul64p(a,b) _mi_mul64(_mi_sr(a,4),_mi_sr(b,4))
#define _mi_6432(a,b)   _mm256_unpacklo_epi32(_mm256_shuffle_epi32(a, shufl), _mm256_shuffle_epi32(b,shufl))
#define _mi_and         _mm256_and_si256
#define _mi_or          _mm256_or_si256
#define _mi_xor         _mm256_xor_si256
#define _mi_andnot      _mm256_andnot_si256
#define _mi_eq          _mm256_cmpeq_epi32
#define _mi_gt          _mm256_cmpgt_epi32
#define _mi_cvtps       _mm256_cvtepi32_ps
#define _m_sum(x) (*(x) + *(x + 1) + *(x + 2) + *(x + 3) + *(x + 4) + *(x + 5) + *(x + 6) + *(x + 7)) //x is float[8]
#define _m_sumd(x) ((double)(*(x)) + *(x + 1) + *(x + 2) + *(x + 3) + *(x + 4) + *(x + 5) + *(x + 6) + *(x + 7)) //x is float[8]
#endif

#define _batch_sz 1024 //batch processing size, see basic.cpp

#if defined WIN32 || defined _WIN32 || defined WINCE
static float __declspec(align(_align_sz)) _align_buf[_m_inc * 2] = { 0.f }; //two __m
#elif defined _linux || defined __linux__
static float __attribute__ ((aligned(_align_sz))) _align_buf[_m_inc * 2] = { 0.f }; 
#else
static __m _align_buf_[2];
static float *_align_buf = (float*)(_align_buf_);
#endif

static const unsigned absmask_32f = 0x7fffffff; //float x; x & absmask_32f = abs(x); Not applicable to int32
static const unsigned signmask_32f = ~absmask_32f; //float x; x & signmask will get the sign bit of x
static const __m absmask_v32f = _m_set1(*(float*)(&absmask_32f));
static const __m signmask_v32f = _m_set1(*(float*)(&signmask_32f));
static const __m _2 = _m_set1(2.f);
static const int shufl = _MM_SHUFFLE(0, 0, 2, 0); //only used in integer multiplication, see "algorithm3.cpp"
#define yuErr(str, ...) { fprintf(stderr, str, __VA_ARGS__), fflush(stderr), throw; } //str is a C string

//=============================== tic-toc ===============================

#if defined WIN32 || defined _WIN32 || defined WINCE

#include <windows.h>

#define _sleep_ Sleep(1000) //1 second

inline double yuTic() {
	LARGE_INTEGER counter;
	QueryPerformanceCounter(&counter);
	return (double)counter.QuadPart;
}

inline double yuTicFreq() {
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	return (double)freq.QuadPart;
}

#elif defined _linux || defined __linux__

#include <unistd.h>

#define _sleep_ sleep(1) //1 second

inline double yuTic() {
	struct timespec tp;
	clock_gettime(CLOCK_MONOTONIC, &tp);
	return (double)tp.tv_sec * 1000000000 + tp.tv_nsec;
}

inline double yuTicFreq() { return 1e9; }

#endif

inline double yuToc(double &tic, const char *str = 0, int show = 1) {
	static double freq = yuTicFreq();
	double toc = yuTic(), tseconds = (toc - tic) / freq;
	if(show == 1) { //in milli-second
		if(str) printf("elapsed time for %s : %dms\n", str, int(tseconds * 1000 + 0.5));
		else printf("elapsed time : %dms\n", int(tseconds * 1000 + 0.5));
	}
	else if(show == 2) { //in second
		if(str) printf("elapsed time for %s : %.1fs\n", str, tseconds);
		else printf("elapsed time : %.1fs\n", tseconds);
	}
	tic = toc;
	return tseconds;
}

//=============================== Shared temporary storage ===============================

struct  Buffer
{
	uchar *data; //pointer to the storage
	int64 sz; //size of the storage, in bytes
	uchar *p; //p is active pointer inside the storage
	int *refcount; //reference count
	void  release() {
		if(refcount) (*refcount)--;
		if(data && sz && !(*refcount))
			delete[] data;
		p = data = 0; sz = 0; refcount = 0;
	}
	void  reserve(int64 n) {
		p = data; //reset p
		if(sz < n) {
			release();
			sz = n;
			p = data = new uchar[sz + sizeof(int)]();
			if(!p)
				yuErr("Buffer: data allocation error!\n");
			refcount = (int*)(p + sz); *refcount = 1;
		} //if(sz >= n), then no change
	}
	~Buffer() { 
		release(); 
	}
	Buffer() {
		p = data = 0; sz = 0; refcount = 0;
	}
	Buffer(int64 n) {
		p = data = 0; sz = 0; refcount = 0;
		reserve(n);
	}
	Buffer(const Buffer &b) { 
		data = b.data; sz = b.sz; p = b.p; refcount = b.refcount;
		if(refcount) (*refcount)++;
	}
	void operator = (const Buffer &b) {
		release();
		data = b.data; sz = b.sz; p = b.p; refcount = b.refcount;
		if(refcount) (*refcount)++;
	}
	uchar * getp() {
		if(p) while(!_m_aligned(p)) p++; //make sure address p is 16/32 bit aligned, crucial for SSE/AVX operations
		if(p && p - data < sz)
			return p;
		else
			yuErr("Buffer::getp(): cannot find appropriate aligned address inside the buffer!\n");
	}
};

//=============================== Packed array (on stack)  ===============================

template<class T, int num> struct AlignedArr
{
	T x[num];
	AlignedArr() { memset(x, 0, num * sizeof(T)); }
	AlignedArr(const AlignedArr *rhs) { memcpy(x, rhs->x, num * sizeof(T)); }
	void operator = (const AlignedArr rhs) { memcpy(x, rhs.x, num * sizeof(T)); }
	T & operator [] (int i) { return x[i]; }
	T operator [] (int i) const { return x[i]; }
	int size() const { return num; }
};
typedef AlignedArr<int, 2> INT2;
typedef AlignedArr<int, 4> INT4;
typedef AlignedArr<float, 2> FLT2;
typedef AlignedArr<float, 4> FLT4;
typedef AlignedArr<double, 2> DBL2;
typedef AlignedArr<double, 4> DBL4;

//=============================== General functions  ===============================

template<typename T>
inline T * getAligned(T *pt) {
	uchar *p = (uchar*)pt;
	if(p) while(!_m_aligned(p)) p++;
	return (T*)p;
}
#ifndef UINT32_MAX
#define UINT32_MAX 0xffffffff
#endif
inline void memCpy(uchar *dst, uchar *src, int64 nBytes) {
	int64 n = 0; //able to copy very large memory
	while(n + UINT32_MAX < nBytes) {
		memcpy(dst, src, UINT32_MAX);
		dst += UINT32_MAX; src += UINT32_MAX; n += UINT32_MAX;		
	}
	memcpy(dst, src, nBytes - n);
}
inline void memSet(void *dst, int val, int64 nBytes) { //val is a 256 bit value (uchar/char)
	char *p = (char*)dst; int64 n = 0; //able to initialize a very large memory
	while(n + UINT32_MAX < nBytes) {
		memset(p, val, UINT32_MAX);
		p += UINT32_MAX; n += UINT32_MAX;
	}
	memset(dst, val, nBytes - n);
}
template<typename T> inline T yumin(T x, T y) { 
	return x < y ? x : y; 
}
template<typename T> inline T yumax(T x, T y) { 
	return x < y ? y : x; 
}
template<typename T> inline T yuabs(T x) { 
	return x < 0 ? -x : x; 
}
inline float yuabsf(float x) {
	union { float f; unsigned i; } vx = { x };
	vx.i &= absmask_32f;
	return vx.f;
}
template<typename T> inline T yusquare(T x) { 
	return x * x;
}
inline int yuround(double x) { //cvRound()
	return _mm_cvtsd_si32(_mm_set_sd(x)); 
}
inline int yuceil(double x) { //cvCeil()
	__m128d _x = _mm_set_sd(x);
	int xi = _mm_cvtsd_si32(_x);
	return xi + _mm_movemask_pd(_mm_cmplt_sd(_mm_cvtsi32_sd(_x, xi), _x));
}
inline int yufloor(double x) {
	__m128d _x = _mm_set_sd(x);
	int xi = _mm_cvtsd_si32(_x);
	return xi - _mm_movemask_pd(_mm_cmplt_sd(_x, _mm_cvtsi32_sd(_x, xi)));
}
inline int yuclip(int x, int a, int b) { 
	return x >= a ? (x < b ? x : b - 1) : a; 
}
inline size_t alignSize(size_t sz, int n) {
	assert(!(n & (n - 1))); //assert n is power of 2
	return (sz + n - 1) & -n;
}
inline float yulog2(float x) { // fast approximate log2(x)
	union { float f; unsigned i; } vx = {x};
	union { unsigned i; float f; } mx = {(vx.i & 0x007FFFFF) | 0x3f000000};
	float y = float(vx.i); y *= 1.1920928955078125e-7f;
	return y - 124.22551499f - 1.498030302f * mx.f - 1.72587999f / (0.3520887068f + mx.f);
}

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

#ifndef COEF_RGB2GRAY
#define COEF_RGB2GRAY {0.299f, 0.587f, 0.114f}
#endif

void  yuSetTo(uchar *dst, const uchar *src, size_t valSz, size_t nVals);
void  yuLSetTo(uchar *dst, const uchar *src, size_t valSz, uint64 nVals);

#define INC 16
#define INCShift 4 //2^4 = 16
#define INCRes(n) ((n) & 15)
#define LOOP_EXECUTE(CMD, ...) { \
/**/  for(n = len >> INCShift; n--; __VA_ARGS__) { \
/**/      CMD(0); CMD(1); CMD(2); CMD(3); CMD(4); CMD(5); CMD(6); CMD(7); \
/**/      CMD(8); CMD(9); CMD(10); CMD(11); CMD(12); CMD(13); CMD(14); CMD(15); } \
/**/  for(n = 0; n < INCRes(len); n++) CMD(n); }

template<typename T>
inline float yuMax(const T *src, int len)
{
	if(len < 1) return 0;
	T m = src[0]; int n;
#define CMD(k) if(m < src[k]) m = src[k];
	LOOP_EXECUTE(CMD, src += INC);
#undef CMD
	return m;
}
template<typename T>
inline T yuMin(const T *src, int len)
{
	if(len < 1) return 0;
	T m = src[0]; int n;
#define CMD(k) if(src[k] < m) m = src[k];
	LOOP_EXECUTE(CMD, src += INC);
#undef CMD
	return m;
}

//dst = dst(mask);
//return the length of the output (nnz in mask).
//keepOrder: if true, then the output data should have exactly the same order as input;
//           if false, output data order might be changed, this is very fast!
template<typename T>
int yuKeepMask(T *dst, const uchar *mask, int len, bool keepOrder)
{
	size_t sz = sizeof(T); int n = 0;
	if(keepOrder) {
		for(; n < len; n++) if(!mask[n]) break;
		if(n == len) return len;
		T *p = dst + n;
		for(n++; n < len; n++) if(mask[n])
			memcpy(p++, dst + n, sz);
		len = (int)(p - dst); return len;
	}
	int m = len; for(n = -1; n < m; ) {
		for(n++; n < m; n++) if(!mask[n]) break; //locate n at the first mask=false		
		for(m--; n < m; m--) if(mask[m]) break; //locate m at the first mask=true in backward order
		memcpy(dst + n, dst + m, sz); //copy dst[m] to dst[n]
	}
	if(n == m && mask[n]) n++; return n;
}
//dst1 = dst1(mask); dst2 = dst2(mask);
template<typename T>
int yuKeepMask(T *dst1, T *dst2, const uchar *mask, int len, bool keepOrder)
{
	size_t sz = sizeof(T); int n = 0;
	if(keepOrder) {
		for(; n < len; n++) if(!mask[n]) break;
		if(n == len) return len;
		T *p1 = dst1 + n, *p2 = dst2 + n;
		for(n++; n < len; n++) if(mask[n]) {
			memcpy(p1++, dst1 + n, sz);
			memcpy(p2++, dst2 + n, sz);
		}
		len = (int)(p1 - dst1); return len;
	}
	int m = len; for(n = -1; n < m; ) {
		for(n++; n < m; n++) if(!mask[n]) break; //locate n at the first mask=false		
		for(m--; n < m; m--) if(mask[m]) break; //locate m at the first mask=true in backward order
		memcpy(dst1 + n, dst1 + m, sz); //copy dst[m] to dst[n]
		memcpy(dst2 + n, dst2 + m, sz);
	}
	if(n == m && mask[n]) n++; return n;
}

//=============================== Functions in basic.cpp  ===============================

void  yuAdd(const float *src1, const float *src2, float *dst, int len);
void  yuSub(const float *src1, const float *src2, float *dst, int len);
void  yuAdd(const int *src1, const int *src2, int *dst, int len);
void  yuSub(const int *src1, const int *src2, int *dst, int len);
void  yuAdd(const double *src1, const double *src2, double *dst, int len);
void  yuAdd(const char *src1, const char *src2, char *dst, int len);
void  yuSub(const char *src1, const char *src2, char *dst, int len);
void  yuAdd(const float *src, float beta, float *dst, int len);
void  yuAdd(const int *src, int beta, int *dst, int len);
void  yuSub(int beta, const int *src, int *dst, int len);
void  yuNeg(const float *src, float *dst, int len);
void  yuSub(float beta, const float *src, float *dst, int len);
void  yuAbs(float *src, float *dst, int len);
void  yuAbsSub(const float *src1, const float *src2, float *dst, int len);
void  yuAbsSub(const float *src, float beta, float *dst, int len);
void  yuAddSubAdd(const float *src1, const float *src2, const float *src3, const float *src4, float *dst, int len);
void  yuAddSubAdd(const int *src1, const int *src2, const int *src3, const int *src4, int *dst, int len);
void  yuSquare(const float *src, float *dst, int len);
void  yuMultiply(const float *src1, const float *src2, float *dst, int len);
void  yuMultiply(const int *src1, const int *src2, int *dst, int len);
void  yuMultiply(const float *src, float alpha, float *dst, int len);
void  yuMultiply(const double *src, double alpha, double *dst, int len);
void  yuInv(const float *src, float *dst, int len);
void  yuDivide(float alpha, const float *src, float *dst, int len);
void  yuScaleAdd(const float *src1, float alpha, const float *src2, float *dst, int len);
void  yuScaleSub(const float *src1, float alpha, const float *src2, float *dst, int len);
void  yuScaleAdd(const double *src1, double alpha, const double *src2, double *dst, int len);
void  yuScaleAdd(const float *src, float alpha, float beta, float *dst, int len);
void  yuAddScale(const float *src1, const float *src2, float alpha, float *dst, int len);
void  yuSubScale(const float *src1, const float *src2, float alpha, float *dst, int len);
void  yuDivAdd(const float *src1, const float *src2, float beta, float *dst, int len);
void  yuSubScaleAdd(const float *src1, const float *src2, float alpha, const float *src3, float *dst, int len);
void  yuScaleDivScaleAdd(const double *src, double gamma, double alpha, double beta, double *dst, int len);
void  yuMulSubMul(const float *src1, const float *src2, const float *src3, const float *src4, float *dst, int len);
void  yuSqSum(const float *src1, const float *src2, float *dst, int len);
void  yuSqrtSqSum(const float *src1, const float *src2, float *dst, int len);
void  yuSqSum(const float *src, float *dst, int len);
void  yuSqrt(const float *src, float *dst, int len);
void  yuRSqrt(const float *src, float *dst, int len);
void  yuMin(const float *src1, const float *src2, float *dst, int len);
void  yuMax(const float *src1, const float *src2, float *dst, int len);
void  yuMin(const float *src, float s, float *dst, int len);
void  yuMax(const float *src, float s, float *dst, int len);
void  yuCompareEq(const char *src1, const char *src2, uchar *dst, int len);
void  yuCompareEq(const char *src, char s, uchar *dst, int len);
void  yuCompareEq(const int *src1, const int *src2, uchar *dst, int len);
void  yuCompareEq(const int *src, int s, uchar *dst, int len);
void  yuCompareLt(const float *src, float s, uchar *dst, int len);
void  yuCompareGt(const float *src, float s, uchar *dst, int len);
void  yuCompareGt(const int *src, int s, uchar *dst, int len);
void  yuAnd(const char *src1, const char *src2, char *dst, int len);
void  yuOr(const char *src1, const char *src2, char *dst, int len);
void  yuAndnot(const char *src1, const char *src2, char *dst, int len);
void  yuAnd(const char *src, char s, char *dst, int len);
void  yuOr(const char *src, char s, char *dst, int len);
void  yuAndnot(const char *src, char s, char *dst, int len);
void  yuMulShift(const int *src1, const int *src2, int *dst, int len, int numBits);
void  yuDetShift(const int *src1, const int *src2, const int *src3, int *dst, int len, int numBits);
void  yuMulSubShift(const int *src1, const int *src2, const int *src3, const int *src4, int *dst, int len, int numBits);
void  yuSortIdx(const int *arr, int *idx, int len);
void  yuSortIdx(const float *arr, int *idx, int len);
void  yuSortIdx(const float *arr, unsigned *idx, unsigned len);

double  yuSum(const float *src, int len);
double  yuDot(const float *src, int len);
double  yuDot(const float *src1, const float *src2, int len);
double  yuDot(const float *src1, const float *src2, const float *src3, int len);
double  yuSum(const double *src, int len);
int  yuNNZ(uchar *src, int len);

void  yuTranspose(const float *src, float *dst, int h, int w, int64 sstep, int64 dstep);
void  yuTranspose(const uchar *src, uchar *dst, int h, int w, int64 sstep, int64 dstep);
void  yuSetTo(uchar *dst, const uchar *src, size_t valSz, size_t nVals);
void  yuLSetTo(uchar *dst, const uchar *src, size_t valSz, uint64 nVals);
void  yuSetMask(uchar *dst, const uchar *mask, uchar val, int len);

void  yuConvert(const int *src, float *dst, int len);
void  yuConvert(const int *src, float scale, float *dst, int len);
void  yuConvert(const float *src, int *dst, int len, bool isTrunc);
void  yuConvert(const float *src, float scale, int *dst, int len, bool isTrunc);
void  yuConvert(const uchar *src, float *dst, int len);
void  yuConvert(const int *src, uchar *dst, int len);