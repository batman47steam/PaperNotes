#include "yustd.h"

/*In this file we define a series of basic functions operating on arrays.
*/

/*------------------------ PART 1 ------------------------
* Functions in Part 1. have the following format:
* (1) Input: 
*     an array;
*	  zero or a couple of other arrays;
*	  zero or a couple of scalars.
* (2) Output:
*	  an array.
*/
//Following is a list of functions defined in Part 1.
//dst = src1 [+;-] src2;
void  yuAdd(const float *src1, const float *src2, float *dst, int len);
void  yuSub(const float *src1, const float *src2, float *dst, int len);
void  yuAdd(const int *src1, const int *src2, int *dst, int len);
void  yuSub(const int *src1, const int *src2, int *dst, int len);
void  yuAdd(const double *src1, const double *src2, double *dst, int len);
void  yuAdd(const char *src1, const char *src2, char *dst, int len);
void  yuSub(const char *src1, const char *src2, char *dst, int len);
//dst = src + beta;
void  yuAdd(const float *src, float beta, float *dst, int len);
void  yuAdd(const int *src, int beta, int *dst, int len);
//dst = beta -src;
void  yuSub(int beta, const int *src, int *dst, int len);
//void  yuNeg(const float *src, float *dst, int len)
void  yuSub(float beta, const float *src, float *dst, int len);
//dst = abs(src;src1-src2;src-beta);
void  yuAbs(float *src, float *dst, int len);
void  yuAbsSub(const float *src1, const float *src2, float *dst, int len);
void  yuAbsSub(const float *src, float beta, float *dst, int len);
//dst = (src1 + src2) - (src3 + src4);
void  yuAddSubAdd(const float *src1, const float *src2, const float *src3, const float *src4, float *dst, int len);
void  yuAddSubAdd(const int *src1, const int *src2, const int *src3, const int *src4, int *dst, int len);
//dst = src .* (src;src2;alpha);
//void  yuSquare(const float *src, float *dst, int len);
void  yuMultiply(const float *src1, const float *src2, float *dst, int len);
void  yuMultiply(const int *src1, const int *src2, int *dst, int len);
void  yuMultiply(const float *src, float alpha, float *dst, int len);
void  yuMultiply(const double *src, double alpha, double *dst, int len);
//dst = alpha ./ src;
//void  yuInv(const float *src, float *dst, int len);
void  yuDivide(float alpha, const float *src, float *dst, int len);
//dst = src1*alpha [+;-] [src2;beta];
void  yuScaleAdd(const float *src1, float alpha, const float *src2, float *dst, int len);
void  yuScaleSub(const float *src1, float alpha, const float *src2, float *dst, int len);
void  yuScaleAdd(const double *src1, double alpha, const double *src2, double *dst, int len);
void  yuScaleAdd(const float *src, float alpha, float beta, float *dst, int len);
//dst = (src1 [+;-] src2) * alpha;
void  yuAddScale(const float *src1, const float *src2, float alpha, float *dst, int len);
void  yuSubScale(const float *src1, const float *src2, float alpha, float *dst, int len);
//dst = src1 ./ (src2 + beta);
void  yuDivAdd(const float *src1, const float *src2, float beta, float *dst, int len);
//dst = (src1 - src2) * alpha + src3;
void  yuSubScaleAdd(const float *src1, const float *src2, float alpha, const float *src3, float *dst, int len);
//dst = gamma * src ./ (alpha * src + beta);
void  yuScaleDivScaleAdd(const double *src, double gamma, double alpha, double beta, double *dst, int len);
//dst = src1 .* src2 - src3 .* [src3;src4];
void  yuMulSubMul(const float *src1, const float *src2, const float *src3, const float *src4, float *dst, int len);
//dst = src1.^2 + src2.^2;
void  yuSqSum(const float *src1, const float *src2, float *dst, int len);
//dst = sqrt(src1.^2 + src2.^2);
void  yuSqrtSqSum(const float *src1, const float *src2, float *dst, int len);
//dst = src(1:2:end).^2 + src(2:2:end).^2; 
void  yuSqSum(const float *src, float *dst, int len);
//dst = [1./] sqrt(src);
void  yuSqrt(const float *src, float *dst, int len);
void  yuRSqrt(const float *src, float *dst, int len);
//dst = [min;max](src1, [src2;s]);
void  yuMin(const float *src1, const float *src2, float *dst, int len);
void  yuMax(const float *src1, const float *src2, float *dst, int len);
void  yuMin(const float *src, float s, float *dst, int len);
void  yuMax(const float *src, float s, float *dst, int len);
//dst = src1 == [src2;s];
void  yuCompareEq(const char *src1, const char *src2, uchar *dst, int len);
void  yuCompareEq(const char *src, char s, uchar *dst, int len);
void  yuCompareEq(const int *src1, const int *src2, uchar *dst, int len);
void  yuCompareEq(const int *src, int s, uchar *dst, int len);
void  yuCompareLt(const float *src, float s, uchar *dst, int len);
void  yuCompareGt(const float *src, float s, uchar *dst, int len);
void  yuCompareGt(const int *src, int s, uchar *dst, int len);
//dst = src1 [and;or;andnot] [src2;s];
void  yuAnd(const char *src1, const char *src2, char *dst, int len);
void  yuOr(const char *src1, const char *src2, char *dst, int len);
void  yuAndnot(const char *src1, const char *src2, char *dst, int len);
void  yuAnd(const char *src, char s, char *dst, int len);
void  yuOr(const char *src, char s, char *dst, int len);
void  yuAndnot(const char *src, char s, char *dst, int len);
//complex integer multiplications
void  yuMulShift(const int *src1, const int *src2, int *dst, int len, int numBits);
void  yuDetShift(const int *src1, const int *src2, const int *src3, int *dst, int len, int numBits);
void  yuMulSubShift(const int *src1, const int *src2, const int *src3, const int *src4, int *dst, int len, int numBits);
//sort array and store the indexes
void  yuSortIdx(const int *arr, int *idx, int len);
void  yuSortIdx(const float *arr, int *idx, int len);
void  yuSortIdx(const float *arr, unsigned *idx, unsigned len);
/*------------------------ PART 2 ------------------------
* Functions in Part 2. have the following format:
* (1) Input:
*     an array;
*	  zero or a couple of other arrays;
*	  zero or a couple of scalars.
* (2) Output:
*	  zero or a couple of scalars.
*/
//Following is a list of functions defined in Part 2.
//sum(src); sum(src.^2); sum(src1.*src2); sum(src1.*src2.*src3);
double  yuSum(const float *src, int len);
//double  yuDot(const float *src, int len);
double  yuDot(const float *src1, const float *src2, int len);
double  yuDot(const float *src1, const float *src2, const float *src3, int len);
double  yuSum(const double *src, int len);
int  yuNNZ(uchar *src, int len);
/*------------------------ PART 3 ------------------------
* Functions in Part 3. have the following features:
* (1) Input:
*     an array;
*	  zero or a couple of other arrays;
*	  zero or a couple of scalars.
* (2) Output:
*	  an array.
* (3) No arithmetic calculation.
*/
//Following is a list of functions defined in Part 3.
//dst = src';
void  yuTranspose(const float *src, float *dst, int h, int w, int64 sstep, int64 dstep);
void  yuTranspose(const uchar *src, uchar *dst, int h, int w, int64 sstep, int64 dstep);
//dst[0~nVals] = val, where val = *src, sizeof(val) = valSz.
void  yuSetTo(uchar *dst, const uchar *src, size_t valSz, size_t nVals);
void  yuLSetTo(uchar *dst, const uchar *src, size_t valSz, uint64 nVals);
//dst(mask) = val;
void  yuSetTo(uchar *dst, const uchar *mask, uchar val, int len);
/*------------------------ PART 4 ------------------------
* Functions in Part 4. are about type conversions.
*/
//Following is a list of functions defined in Part 4.
//cast int to float
//void  yuConvert(const int *src, float *dst, int len);
//cast int to float, scale it afterward
void  yuConvert(const int *src, float scale, float *dst, int len);
//cast float to int, truncation or round to
//void  yuConvert(const float *src, int *dst, int len, bool isTrunc);
//cast float to int, scale the float beforehand
void  yuConvert(const float *src, float scale, int *dst, int len, bool isTrunc);
//uchar to float
void  yuConvert(const uchar *src, float *dst, int len);
//int to uchar
void  yuConvert(const int *src, uchar *dst, int len);







//==============================================================================P1
///dst = src1 + src2; specialized for src1==src2 or src1==src2==dst.
void  yuAdd(const float *src1, const float *src2, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len);
	if(src1 == src2) {
		__m _v;
		if(src1 == dst) {
			for(; len-- && !_m_aligned(dst); *dst++ += *src1++);
			for(n = _m_nsse(++len); n--; dst += _m_inc) {
				_v = _m_load(dst);
				_m_store(dst, _m_add(_v, _v));
			}
			src1 = src2 = dst;
		}
		else {
			if(isAligned) for(; n--; src1 += _m_inc, dst += _m_inc) {
				_v = _m_load(src1);
				_m_store(dst, _m_add(_v, _v));
			}
			else for(; n--; src1 += _m_inc, dst += _m_inc) {
				_v = _m_loadu(src1);
				_m_storeu(dst, _m_add(_v, _v));
			}
			src2 = src1;
		}
	}
	else {
		if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
			_m_store(dst, _m_add(_m_load(src1), _m_load(src2)));
		else for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
			_m_storeu(dst, _m_add(_m_loadu(src1), _m_loadu(src2)));
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = src1[n] + src2[n];
}
///dst = src1 - src2;
void  yuSub(const float *src1, const float *src2, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_store(dst, _m_sub(_m_load(src1), _m_load(src2)));
	else for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_storeu(dst, _m_sub(_m_loadu(src1), _m_loadu(src2)));
	for(n = 0; n < res; n++)
		dst[n] = src1[n] - src2[n];
}
///dst = src1 + src2;
void  yuAdd(const int *src1, const int *src2, int *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__mi *ps1 = (__mi*)src1, *ps2 = (__mi*)src2, *pd = (__mi*)dst;
	if(isAligned) while(n--)
		_mi_store(pd++, _mi_add(_mi_load(ps1++), _mi_load(ps2++)));
	else while(n--)
		_mi_storeu(pd++, _mi_add(_mi_loadu(ps1++), _mi_loadu(ps2++)));
	if(res) {
		src1 = (int*)ps1, src2 = (int*)ps2, dst = (int*)pd;
		for(n = 0; n < res; n++)
			dst[n] = src1[n] + src2[n];
	}
}
//dst = src1 - src2;
void  yuSub(const int *src1, const int *src2, int *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__mi *_src1 = (__mi*)src1, *_src2 = (__mi*)src2, *_dst = (__mi*)dst;
	if(isAligned) while(n--)
		_mi_store(_dst++, _mi_sub(_mi_load(_src1++), _mi_load(_src2++)));
	else while(n--)
		_mi_storeu(_dst++, _mi_sub(_mi_loadu(_src1++), _mi_loadu(_src2++)));
	src1 = (int*)_src1, src2 = (int*)_src2, dst = (int*)_dst;
	for(n = 0; n < res; n++)
		dst[n] = src1[n] - src2[n];
}
///dst = src1 + src2; double version.
void  yuAdd(const double *src1, const double *src2, double *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
#ifdef USE_AVX
	int n = len >> 2, res = len & 3; //int n = len / 4, res = len % 4;
	if(isAligned) for(; n--; src1 += 4, src2 += 4, dst += 4)
		_mm256_store_pd(dst, _mm256_add_pd(_mm256_load_pd(src1), _mm256_load_pd(src2)));
	else for(; n--; src1 += 4, src2 += 4, dst += 4)
		_mm256_storeu_pd(dst, _mm256_add_pd(_mm256_loadu_pd(src1), _mm256_loadu_pd(src2)));
#else
	int n = len / 2, res = len % 2;
	if(isAligned) for(; n--; src1 += 2, src2 += 2, dst += 2)
		_mm_store_pd(dst, _mm_add_pd(_mm_load_pd(src1), _mm_load_pd(src2)));
	else for(; n--; src1 += 2, src2 += 2, dst += 2)
		_mm_storeu_pd(dst, _mm_add_pd(_mm_loadu_pd(src1), _mm_loadu_pd(src2)));
#endif
	for(n = 0; n < res; n++)
		dst[n] = src1[n] + src2[n];
}
///dst = src1 + src2; char version.
void  yuAdd(const char *src1, const char *src2, char *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	__mi *_src1 = (__mi*)src1, *_src2 = (__mi*)src2, *_dst = (__mi*)dst;
#ifdef USE_AVX
	int n = len >> 5, res = len & 31; //int n = len / 32, res = len % 32;
	if(isAligned) while(n--)
		_mi_store(_dst++, _mm256_add_epi8(_mi_load(_src1++), _mi_load(_src2++)));
	else while(n--)
		_mi_storeu(_dst++, _mm256_add_epi8(_mi_loadu(_src1++), _mi_loadu(_src2++)));
#else
	int n = len >> 4, res = len & 15; //int n = len / 16, res = len % 16;
	if(isAligned) while(n--)
		_mi_store(_dst++, _mm_add_epi8(_mi_load(_src1++), _mi_load(_src2++)));
	else while(n--)
		_mi_storeu(_dst++, _mm_add_epi8(_mi_loadu(_src1++), _mi_loadu(_src2++)));
#endif
	if(res) {
		src1 = (char*)_src1, src2 = (char*)_src2, dst = (char*)_dst;
		for(n = 0; n < res; n++)
			dst[n] = src1[n] + src2[n];
	}
}
///dst = src1 - src2; char version.
void  yuSub(const char *src1, const char *src2, char *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	__mi *_src1 = (__mi*)src1, *_src2 = (__mi*)src2, *_dst = (__mi*)dst;
#ifdef USE_AVX
	int n = len >> 5, res = len & 31; //int n = len / 32, res = len % 32;
	if(isAligned) while(n--)
		_mi_store(_dst++, _mm256_sub_epi8(_mi_load(_src1++), _mi_load(_src2++)));
	else while(n--)
		_mi_storeu(_dst++, _mm256_sub_epi8(_mi_loadu(_src1++), _mi_loadu(_src2++)));
#else
	int n = len >> 4, res = len & 15; //int n = len / 16, res = len % 16;
	if(isAligned) while(n--)
		_mi_store(_dst++, _mm_sub_epi8(_mi_load(_src1++), _mi_load(_src2++)));
	else while(n--)
		_mi_storeu(_dst++, _mm_sub_epi8(_mi_loadu(_src1++), _mi_loadu(_src2++)));
#endif
	if(res) {
		src1 = (char*)_src1, src2 = (char*)_src2, dst = (char*)_dst;
		for(n = 0; n < res; n++)
			dst[n] = src1[n] - src2[n];
	}
}
//==============================================================================P1
///dst = src + beta;
void  yuAdd(const float *src, float beta, float *dst, int len)
{
	_m_clear;
	__m _beta = _m_set1(beta); int n;
	if(src == dst) {
		for(; len-- && !_m_aligned(dst); *dst++ += beta);
		for(n = _m_nsse(++len); n--; dst += _m_inc)
			_m_store(dst, _m_add(_m_load(dst), _beta));
		src = dst;
	}
	else {
		bool isAligned = _m_aligned(size_t(src) | size_t(dst));
		if(isAligned) for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_store(dst, _m_add(_m_load(src), _beta));
		else for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_storeu(dst, _m_add(_m_loadu(src), _beta));
	}
	for(n = _m_res(len); n--; *dst++ = *src++ + beta);
}
///dst = src + beta
void  yuAdd(const int *src, int beta, int *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__mi *ps = (__mi*)src, *pd = (__mi*)dst, _beta = _mi_set1(beta);
	if(isAligned) while(n--)
		_mi_store(pd++, _mi_add(_mi_load(ps++), _beta));
	else while(n--)
		_mi_storeu(pd++, _mi_add(_mi_loadu(ps++), _beta));
	if(res) {
		src = (int*)ps, dst = (int*)pd;
		for(n = 0; n < res; n++)
			dst[n] = src[n] + beta;
	}
}
//==============================================================================P1
//dst = beta - src;
void  yuSub(int beta, const int *src, int *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__mi *ps = (__mi*)src, *pd = (__mi*)dst, _beta = _mi_set1(beta);
	if(isAligned) while(n--)
		_mi_store(pd++, _mi_sub(_beta, _mi_load(ps++)));
	else while(n--)
		_mi_storeu(pd++, _mi_add(_beta, _mi_loadu(ps++)));
	src = (int*)ps, dst = (int*)pd;
	for(n = 0; n < res; n++)
		dst[n] = beta - src[n];
}
//dst = -src;
//Only for float32 data. Integer values are stored as ²¹Âë, more complicate.
//The neg() is just to set the sign bit by the opposite value.
//For integer values, please use yuSub(0,src,dst) instead.
void  yuNeg(const float *src, float *dst, int len)
{
	_m_clear;
	__m _v; int n;
	//(~x&signmask)|(x&absmask_32f) will result in -x
	if(src == dst) {
		for(n = 0; n < len && !_m_aligned(dst + n); n++)
			dst[n] = -dst[n];
		if(n) dst += n, len -= n;
		for(n = _m_nsse(len); n--; dst += _m_inc) {
			_v = _m_load(dst);
			_m_store(dst, _m_or(_m_andnot(_v, signmask_v32f), _m_and(_v, absmask_v32f)));
		}
		src = dst;
	}
	else {
		bool isAligned = _m_aligned(size_t(src) | size_t(dst));
		if(isAligned) for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc) {
			_v = _m_load(src);
			_m_store(dst, _m_or(_m_andnot(_v, signmask_v32f), _m_and(_v, absmask_v32f)));
		}
		else for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc) {
			_v = _m_loadu(src);
			_m_storeu(dst, _m_or(_m_andnot(_v, signmask_v32f), _m_and(_v, absmask_v32f)));
		}
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = -src[n];
}
//dst = beta - src;
void  yuSub(float beta, const float *src, float *dst, int len)
{
	if(yuabs(beta) < 1e-16f)
		return yuNeg(src, dst, len);
	_m_clear;
	__m _beta = _m_set1(beta); int n;
	//(~x&signmask)|(x&absmask_32f) will result in -x
	if(src == dst) {
		for(n = 0; n < len && !_m_aligned(dst + n); n++)
			dst[n] = beta - dst[n];
		if(n) dst += n, len -= n;
		for(n = _m_nsse(len); n--; dst += _m_inc)
			_m_store(dst, _m_sub(_beta, _m_load(dst)));
		src = dst;
	}
	else {
		bool isAligned = _m_aligned(size_t(src) | size_t(dst));
		if(isAligned) for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_store(dst, _m_sub(_beta, _m_load(src)));
		else for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_storeu(dst, _m_sub(_beta, _m_loadu(src)));
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = beta - src[n];
}
//==============================================================================P1
//dst = abs(src);
//For float data, the highest bit is the sign bit.
//The abs operation is just to set the sign bit to 0.
void  yuAbs(float *src, float *dst, int len)
{
	_m_clear;
	int n;
	if(src == dst) {
		for(n = 0; n < len && !_m_aligned(dst + n); n++)
			dst[n] = yuabsf(dst[n]);
		dst += n; len -= n;
		for(n = _m_nsse(len); n--; dst += _m_inc)
			_m_store(dst, _m_and(absmask_v32f, _m_load(dst)));
		src = dst;
	}
	else { 
		bool isAligned = _m_aligned(size_t(src) | size_t(dst));
		if(isAligned) for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_store(dst, _m_and(absmask_v32f, _m_load(src)));
		else for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_storeu(dst, _m_and(absmask_v32f, _m_loadu(src)));
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = yuabsf(src[n]);
}
///dst = abs(src1 - src2);
void  yuAbsSub(const float *src1, const float *src2, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	//unsigned absmask_32f = 0x7fffffff;
	//__m absmask_v32f = _m_set1(*(float*)(&absmask_32f));
	if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_store(dst, _m_and(absmask_v32f, _m_sub(_m_load(src1), _m_load(src2))));
	else for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_storeu(dst, _m_and(absmask_v32f, _m_sub(_m_loadu(src1), _m_loadu(src2))));
	for(n = 0; n < res; n++)
		dst[n] = yuabsf(src1[n] - src2[n]);
}
///dst = abs(src - beta);
void  yuAbsSub(const float *src, float beta, float *dst, int len)
{
	_m_clear;
	__m _beta = _m_set1(beta); int n;
	if(src == dst) {
		for(n = 0; n < len && !_m_aligned(dst + n); n++)
			dst[n] = yuabsf(dst[n] - beta);
		if(n) dst += n, len -= n;
		for(n = _m_nsse(len); n--; dst += _m_inc)
			_m_store(dst, _m_and(absmask_v32f, _m_sub(_m_load(dst), _beta)));
		src = dst;
	}
	else {
		bool isAligned = _m_aligned(size_t(src) | size_t(dst));
		if(isAligned) for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_store(dst, _m_and(absmask_v32f, _m_sub(_m_load(src), _beta)));
		else for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_storeu(dst, _m_and(absmask_v32f, _m_sub(_m_loadu(src), _beta)));
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = yuabsf(src[n] - beta);
}
void  yuAddSubAdd(const float *src1, const float *src2, const float *src3, const float *src4, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(src3) | size_t(src4) | size_t(dst));
	int n;
	if(isAligned) for(n = _m_nsse(len); n--; src1 += _m_inc, src2 += _m_inc, src3 += _m_inc, src4 += _m_inc, dst += _m_inc)
		_m_store(dst, _m_sub(_m_add(_m_load(src1), _m_load(src2)), _m_add(_m_load(src3), _m_load(src4))));
	for(n = _m_nsse(len); n--; src1 += _m_inc, src2 += _m_inc, src3 += _m_inc, src4 += _m_inc, dst += _m_inc)
		_m_storeu(dst, _m_sub(_m_add(_m_loadu(src1), _m_loadu(src2)), _m_add(_m_loadu(src3), _m_loadu(src4))));
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = (src1[n] + src2[n]) - (src3[n] + src4[n]);
}
void  yuAddSubAdd(const int *src1, const int *src2, const int *src3, const int *src4, int *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(src3) | size_t(src4) | size_t(dst));
	__mi *p1 = (__mi*)src1, *p2 = (__mi*)src2, *p3 = (__mi*)src3, *p4 = (__mi*)src4, *pd = (__mi*)dst;
	int n;
	if(isAligned) for(n = _m_nsse(len); n--;)
		_mi_store(pd++, _mi_sub(_mi_add(_mi_load(p1++), _mi_load(p2++)), _mi_add(_mi_load(p3++), _mi_load(p4++))));
	for(n = _m_nsse(len); n--;)
		_mi_storeu(pd++, _mi_sub(_mi_add(_mi_loadu(p1++), _mi_loadu(p2++)), _mi_add(_mi_loadu(p3++), _mi_loadu(p4++)))); 
	if(len = _m_res(len)) {
		src1 = (int*)p1, src2 = (int*)p2, src3 = (int*)p3, src4 = (int*)p4, dst = (int*)pd;
		for(n = 0; n < len; n++)
			dst[n] = (src1[n] + src2[n]) - (src3[n] + src4[n]);
	}
}

//==============================================================================P1
//dst = src.^2;
void  yuSquare(const float *src, float *dst, int len)
{
	_m_clear;
	__m _v; int n;
	if(src == dst) {
		for(n = 0; n < len && !_m_aligned(dst + n); n++)
			dst[n] *= dst[n];
		if(n) dst += n, len -= n;
		for(n = _m_nsse(len); n--; dst += _m_inc) {
			_v = _m_load(dst);
			_m_store(dst, _m_mul(_v, _v));
		}
		src = dst;
	}
	else {
		bool isAligned = _m_aligned(size_t(src) | size_t(dst));
		if(isAligned) for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc) {
			_v = _m_load(src);
			_m_store(dst, _m_mul(_v, _v));
		}
		else for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc) {
			_v = _m_loadu(src);
			_m_storeu(dst, _m_mul(_v, _v));
		}
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = src[n] * src[n];
}
///dst = src1 .* src2;
void  yuMultiply(const float *src1, const float *src2, float *dst, int len)
{
	if(src1 == src2)
		return yuSquare(src1, dst, len);
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_store(dst, _m_mul(_m_load(src1), _m_load(src2)));
	else for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_storeu(dst, _m_mul(_m_loadu(src1), _m_loadu(src2)));
	for(n = 0; n < res; n++)
		dst[n] = src1[n] * src2[n];
}
//dst = src1 .* src2;
//Note: about the same speed as float*float;
void  yuMultiply(const int *src1, const int *src2, int *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__mi *_src1 = (__mi*)src1, *_src2 = (__mi*)src2, *_dst = (__mi*)dst;
	if(isAligned) while(n--)
		_mi_store(_dst++, _mi_mul(_mi_load(_src1++), _mi_load(_src2++)));
	else while(n--)
		_mi_storeu(_dst++, _mi_mul(_mi_loadu(_src1++), _mi_loadu(_src2++)));
	if(res) {
		src1 = (int*)_src1, src2 = (int*)_src2, dst = (int*)_dst;
		for(n = 0; n < res; n++)
			dst[n] = src1[n] * src2[n];
	}
}
///dst = src * alpha;
void  yuMultiply(const float *src, float alpha, float *dst, int len)
{
	_m_clear;
	__m _alpha = _m_set1(alpha); int n;
	if(src == dst) { 
		for(; len-- && !_m_aligned(dst); *dst++ *= alpha);
		for(n = _m_nsse(++len); n--; dst += _m_inc)
			_m_store(dst, _m_mul(_m_load(dst), _alpha));
		src = dst;
	}
	else { 
		bool isAligned = _m_aligned(size_t(src) | size_t(dst));
		if(isAligned) for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_store(dst, _m_mul(_m_load(src), _alpha));
		else for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_storeu(dst, _m_mul(_m_loadu(src), _alpha));
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = src[n] * alpha;
}
///dst = src * alpha; double version.
void  yuMultiply(const double *src, double alpha, double *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src) | size_t(dst));
#ifdef USE_AVX
	int n = len >> 2, res = len & 3; //int n = len / 4, res = len % 4;
	__m256d _alpha = _mm256_set1_pd(alpha);
	if(isAligned) for(; n--; src += 4, dst += 4)
		_mm256_store_pd(dst, _mm256_mul_pd(_mm256_load_pd(src), _alpha));
	else for(; n--; src += 4, dst += 4)
		_mm256_storeu_pd(dst, _mm256_mul_pd(_mm256_loadu_pd(src), _alpha));
#else
	int n = len / 2, res = len % 2;
	__m128d _alpha = _mm_set1_pd(alpha);
	if(isAligned) for(; n--; src += 2, dst += 2)
		_mm_store_pd(dst, _mm_mul_pd(_mm_load_pd(src), _alpha));
	else for(; n--; src += 2, dst += 2)
		_mm_storeu_pd(dst, _mm_mul_pd(_mm_loadu_pd(src), _alpha));
#endif
	for(n = 0; n < res; n++)
		dst[n] = src[n] * alpha;
}
//==============================================================================P1
//dst = 1/src; dst(src==0)=0;
void  yuInv(const float *src, float *dst, int len)
{
	_m_clear;
	////! Implementation 1. less accurate!
	//bool isAligned = _m_aligned(size_t(src) | size_t(dst));
	//int n = _m_nsse(len), res = _m_res(len);
	//__m _src;
	//if(isAligned) for(; n--; src += _m_inc, dst += _m_inc) {
	//	_src = _m_load(src);
	//	_m_store(dst, _m_and(_m_inv(_src), _m_neq(_src, _m_zero())));
	//}
	//else for(; n--; src += _m_inc, dst += _m_inc) {
	//	_src = _m_loadu(src);
	//	_m_storeu(dst, _m_and(_m_inv(_src), _m_neq(_src, _m_zero())));
	//}
	//! Implementation 2. accurate!
	int n = len >> 3, res = len & 7; //int n = len / 8, res = len % 8;
	for(; n--; src += 8, dst += 8) {
		dst[0] = src[0] ? 1.f / src[0] : 0.f;
		dst[1] = src[1] ? 1.f / src[1] : 0.f;
		dst[2] = src[2] ? 1.f / src[2] : 0.f;
		dst[3] = src[3] ? 1.f / src[3] : 0.f;
		dst[4] = src[4] ? 1.f / src[4] : 0.f;
		dst[5] = src[5] ? 1.f / src[5] : 0.f;
		dst[6] = src[6] ? 1.f / src[6] : 0.f;
		dst[7] = src[7] ? 1.f / src[7] : 0.f;
	}
	for(n = 0; n < res; n++)
		dst[n] = src[n] ? 1.f / src[n] : 0.f;
}
///dst = alpha / src; dst(src==0) = 0;
void  yuDivide(float alpha, const float *src, float *dst, int len)
{
	if(yuabs(alpha - 1.f) < 1e-16f)
		return yuInv(src, dst, len);
	_m_clear;
	__m _alpha = _m_set1(alpha), _v; int n;
	if(src == dst) {
		for(n = 0; n < len && !_m_aligned(dst + n); n++)
			dst[n] = dst[n] ? alpha / dst[n] : 0.f;
		if(n) dst += n, len -= n;
		for(n = _m_nsse(len); n--; dst += _m_inc) {
			_v = _m_load(dst);
			_m_store(dst, _m_and(_m_div(_alpha, _v), _m_neq(_v, _m_zero())));
		}
		src = dst;
	}
	else {
		bool isAligned = _m_aligned(size_t(src) | size_t(dst));
		if(isAligned) for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc) {
			_v = _m_load(src);
			_m_store(dst, _m_and(_m_div(_alpha, _v), _m_neq(_v, _m_zero())));
		}
		else for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc) {
			_v = _m_loadu(src);
			_m_storeu(dst, _m_and(_m_div(_alpha, _v), _m_neq(_v, _m_zero())));
		}
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = src[n] ? alpha / src[n] : 0.f;
}
//==============================================================================P1
///dst = src1 * alpha + src2;
void  yuScaleAdd(const float *src1, float alpha, const float *src2, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__m _alpha = _m_set1(alpha);
	if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_store(dst, _m_add(_m_mul(_m_load(src1), _alpha), _m_load(src2)));
	else for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_storeu(dst, _m_add(_m_mul(_m_loadu(src1), _alpha), _m_loadu(src2)));
	for(n = 0; n < res; n++)
		dst[n] = src1[n] * alpha + src2[n];
}
///dst = src1 * alpha - src2;
void  yuScaleSub(const float *src1, float alpha, const float *src2, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__m _alpha = _m_set1(alpha);
	if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_store(dst, _m_sub(_m_mul(_m_load(src1), _alpha), _m_load(src2)));
	else for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_store(dst, _m_sub(_m_mul(_m_load(src1), _alpha), _m_load(src2)));
	for(n = 0; n < res; n++)
		dst[n] = src1[n] * alpha - src2[n];
}
///dst = src1 * alpha + src2; double version.
void  yuScaleAdd(const double *src1, double alpha, const double *src2, double *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
#ifdef USE_AVX
	int n = len >> 2, res = len & 3; //int n = len / 4, res = len % 4;
	__m256d _alpha = _mm256_set1_pd(alpha);
	if(isAligned) for(; n--; src1 += 4, src2 += 4, dst += 4)
		_mm256_store_pd(dst, _mm256_add_pd(_mm256_mul_pd(_mm256_load_pd(src1), _alpha), _mm256_load_pd(src2)));
	else for(; n--; src1 += 4, src2 += 4, dst += 4)
		_mm256_storeu_pd(dst, _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(src1), _alpha), _mm256_loadu_pd(src2)));
#else
	int n = len / 2, res = len % 2;
	__m128d _alpha = _mm_set1_pd(alpha);
	if(isAligned) for(; n--; src1 += 2, src2 += 2, dst += 2)
		_mm_store_pd(dst, _mm_add_pd(_mm_mul_pd(_mm_load_pd(src1), _alpha), _mm_load_pd(src2)));
	else for(; n--; src1 += 2, src2 += 2, dst += 2)
		_mm_storeu_pd(dst, _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(src1), _alpha), _mm_loadu_pd(src2)));
#endif
	for(n = 0; n < res; n++)
		dst[n] = src1[n] * alpha + src2[n];
}
///dst = src * alpha + beta;
void  yuScaleAdd(const float *src, float alpha, float beta, float *dst, int len)
{
	_m_clear;
	__m _alpha = _m_set1(alpha), _beta = _m_set1(beta); int n;
	if(src == dst) {
		for(n = 0; n < len && !_m_aligned(dst + n); n++)
			dst[n] = dst[n] * alpha + beta;
		if(n) dst += n, len -= n;
		for(n = _m_nsse(len); n--; dst += _m_inc)
			_m_store(dst, _m_add(_m_mul(_m_load(dst), _alpha), _beta));
		src = dst;
	}
	else {
		bool isAligned = _m_aligned(size_t(src) | size_t(dst));
		if(isAligned) for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_store(dst, _m_add(_m_mul(_m_load(src), _alpha), _beta));
		else for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_storeu(dst, _m_add(_m_mul(_m_loadu(src), _alpha), _beta));
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = src[n] * alpha + beta;
}
//==============================================================================P1
///dst = (src1 + src2) * alpha;
void  yuAddScale(const float *src1, const float *src2, float alpha, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__m _alpha = _m_set1(alpha);
	if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_store(dst, _m_mul(_m_add(_m_load(src1), _m_load(src2)), _alpha));
	else for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_storeu(dst, _m_mul(_m_add(_m_loadu(src1), _m_loadu(src2)), _alpha));
	for(n = 0; n < res; n++)
		dst[n] = (src1[n] + src2[n]) * alpha;
}
///dst = (src1 - src2) * alpha;
void  yuSubScale(const float *src1, const float *src2, float alpha, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__m _alpha = _m_set1(alpha);
	if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_store(dst, _m_mul(_m_sub(_m_load(src1), _m_load(src2)), _alpha));
	else for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_storeu(dst, _m_mul(_m_sub(_m_loadu(src1), _m_loadu(src2)), _alpha));
	for(n = 0; n < res; n++)
		dst[n] = (src1[n] - src2[n]) * alpha;
}
//==============================================================================P1
///dst = src1 ./ (src2 + beta);
void  yuDivAdd(const float *src1, const float *src2, float beta, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__m _beta = _m_set1(beta);
	if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_store(dst, _m_div(_m_load(src1), _m_add(_m_load(src2), _beta)));
	else for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_storeu(dst, _m_div(_m_loadu(src1), _m_add(_m_loadu(src2), _beta)));
	for(n = 0; n < res; n++)
		dst[n] = src1[n] / (src2[n] + beta);
}
//==============================================================================P1
///dst = (src1 - src2) * alpha + src3;
void  yuSubScaleAdd(const float *src1, const float *src2, float alpha, const float *src3, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(src3) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__m _alpha = _m_set1(alpha);
	if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, src3 += _m_inc, dst += _m_inc)
		_m_store(dst, _m_add(_m_mul(_m_sub(_m_load(src1), _m_load(src2)), _alpha), _m_load(src3)));
	else for(; n--; src1 += _m_inc, src2 += _m_inc, src3 += _m_inc, dst += _m_inc)
		_m_storeu(dst, _m_add(_m_mul(_m_sub(_m_loadu(src1), _m_loadu(src2)), _alpha), _m_loadu(src3)));
	for(n = 0; n < res; n++)
		dst[n] = (src1[n] - src2[n]) * alpha + src3[n];
}
//==============================================================================P1
///dst = gamma * src ./ (alpha * src + beta);
///For double precision;
///dst = src * alpha; double version.
///We won't check whether the denominator is ZERO!
///This function has not been made a yu::Mat wrapper yet cause it's too complicate.
void  yuScaleDivScaleAdd(const double *src, double gamma, double alpha, double beta, double *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src) | size_t(dst));
#ifdef USE_AVX
	int n = len >> 2, res = len & 3; //int n = len / 4, res = len % 4;
	__m256d _alpha = _mm256_set1_pd(alpha);
	__m256d _beta = _mm256_set1_pd(beta);
	__m256d _gamma = _mm256_set1_pd(gamma);
	__m256d _src;
	if(isAligned) for(; n--; src += 4, dst += 4) {
		_src = _mm256_load_pd(src);
		_mm256_store_pd(dst, _mm256_div_pd(_mm256_mul_pd(_src, _gamma), _mm256_add_pd(_mm256_mul_pd(_src, _alpha), _beta)));
	}
	else for(; n--; src += 4, dst += 4) {
		_src = _mm256_loadu_pd(src);
		_mm256_storeu_pd(dst, _mm256_div_pd(_mm256_mul_pd(_src, _gamma), _mm256_add_pd(_mm256_mul_pd(_src, _alpha), _beta)));
	}
#else
	int n = len / 2, res = len % 2;
	__m128d _alpha = _mm_set1_pd(alpha);
	__m128d _beta = _mm_set1_pd(beta);
	__m128d _gamma = _mm_set1_pd(gamma);
	__m128d _src;
	if(isAligned) for(; n--; src += 2, dst += 2) {
		_src = _mm_load_pd(src);
		_mm_store_pd(dst, _mm_div_pd(_mm_mul_pd(_src, _gamma), _mm_add_pd(_mm_mul_pd(_src, _alpha), _beta)));
	}
	else for(; n--; src += 2, dst += 2) {
		_src = _mm_loadu_pd(src);
		_mm_storeu_pd(dst, _mm_div_pd(_mm_mul_pd(_src, _gamma), _mm_add_pd(_mm_mul_pd(_src, _alpha), _beta)));
	}
#endif
	for(n = 0; n < res; n++)
		dst[n] = src[n] * gamma / (src[n] * alpha + beta);
}
//==============================================================================P1
///dst = src1 .* src2 - src3 .* src4;
void  yuMulSubMul(const float *src1, const float *src2, const float *src3, const float *src4, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(src3) | size_t(src4) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	if(src3 == src4) {
		__m _tmp;
		if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, src3 += _m_inc, dst += _m_inc) {
			_tmp = _m_load(src3);
			_m_store(dst, _m_sub(_m_mul(_m_load(src1), _m_load(src2)), _m_mul(_tmp, _tmp)));
		}
		else for(; n--; src1 += _m_inc, src2 += _m_inc, src3 += _m_inc, dst += _m_inc) {
			_tmp = _m_loadu(src3);
			_m_storeu(dst, _m_sub(_m_mul(_m_loadu(src1), _m_loadu(src2)), _m_mul(_tmp, _tmp)));
		}
		src4 = src3;
	}
	else {
		if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, src3 += _m_inc, src4 += _m_inc, dst += _m_inc)
			_m_store(dst, _m_sub(_m_mul(_m_load(src1), _m_load(src2)), _m_mul(_m_load(src3), _m_load(src4))));
		else for(; n--; src1 += _m_inc, src2 += _m_inc, src3 += _m_inc, src4 += _m_inc, dst += _m_inc)
			_m_storeu(dst, _m_sub(_m_mul(_m_loadu(src1), _m_loadu(src2)), _m_mul(_m_loadu(src3), _m_loadu(src4))));
	}
	for(n = 0; n < res; n++)
		dst[n] = src1[n] * src2[n] - src3[n] * src4[n];
}
//==============================================================================P1
///dst = src1.^2 + src2.^2;
void  yuSqSum(const float *src1, const float *src2, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	__m _v1, _v2; int n = _m_nsse(len);
	if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc) {
		_v1 = _m_load(src1), _v2 = _m_load(src2);
		_m_store(dst, _m_add(_m_mul(_v1, _v1), _m_mul(_v2, _v2)));
	}
	else for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc) {
		_v1 = _m_loadu(src1), _v2 = _m_loadu(src2);
		_m_storeu(dst, _m_add(_m_mul(_v1, _v1), _m_mul(_v2, _v2)));
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = src1[n] * src1[n] + src2[n] * src2[n];
}
//==============================================================================P1
///dst = sqrt(src1.^2 + src2.^2);
void  yuSqrtSqSum(const float *src1, const float *src2, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	__m _v1, _v2; int n = _m_nsse(len);
	if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc) {
		_v1 = _m_load(src1), _v2 = _m_load(src2);
		_m_store(dst, _m_sqrt(_m_add(_m_mul(_v1, _v1), _m_mul(_v2, _v2))));
	}
	else for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc) {
		_v1 = _m_loadu(src1), _v2 = _m_loadu(src2);
		_m_storeu(dst, _m_sqrt(_m_add(_m_mul(_v1, _v1), _m_mul(_v2, _v2))));
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = sqrtf(src1[n] * src1[n] + src2[n] * src2[n]);
}
//==============================================================================P1
///src: float[len*2]; dst: float[len];
///dst = src(1:2:end).^2 + src(2:2:end).^2;
void  yuSqSum(const float *src, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src));
	__m _v; int n = _m_nsse(len);
	float *p = _align_buf, *q = _align_buf + _m_inc;
#define CMD1 dst[0]=p[0]+p[1]; dst[1]=p[2]+p[3]; dst[2]=p[4]+p[5]; dst[3]=p[6]+p[7]
#ifdef USE_AVX
#define CMD CMD1; dst[4]=q[0]+q[1]; dst[5]=q[2]+q[3]; dst[6]=q[4]+q[5]; dst[7]=q[6]+q[7]
#else
#define CMD CMD1
#endif
	if(isAligned) for(; n--; dst += _m_inc) {
		_v = _m_load(src); src += _m_inc;
		_m_store(p, _m_mul(_v, _v));
		_v = _m_load(src); src += _m_inc;
		_m_store(q, _m_mul(_v, _v));
		CMD;
	}
	else for(; n--; dst += _m_inc) {
		_v = _m_loadu(src); src += _m_inc;
		_m_store(p, _m_mul(_v, _v));
		_v = _m_loadu(src); src += _m_inc;
		_m_store(q, _m_mul(_v, _v));
		CMD;
	}
	for(n = _m_res(len); n--; src += 2)
		*dst++ = src[0] * src[0] + src[1] * src[1];
#undef CMD1
#undef CMD
}
//==============================================================================P1
///dst = sqrt(src);
///Warning: will produce NAN if there are negative values in src!
void  yuSqrt(const float *src, float *dst, int len)
{
	_m_clear;
	int n;
	if(src == dst) {
		for(n = 0; n < len && !_m_aligned(dst + n); n++)
			dst[n] = sqrtf(dst[n]);
		if(n) dst += n, len -= n;
		for(n = _m_nsse(len); n--; dst += _m_inc)
			_m_store(dst, _m_sqrt(_m_load(dst)));
		src = dst;
	}
	else { 
		bool isAligned = _m_aligned(size_t(src) | size_t(dst));
		if(isAligned) for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_store(dst, _m_sqrt(_m_load(src)));
		for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_storeu(dst, _m_sqrt(_m_loadu(src)));
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = sqrtf(src[n]);
}
//dst = 1 / sqrt(src);
//Notice: won't produce NAN or INF results cause we check whether src is less than eps.
//Warning: the _rsqrt function of AVX/SSE loses a lot of precision!!!	
void  yuRSqrt(const float *src, float *dst, int len)
{	
	//! Implementation 1. less accurate! safe!
	_m_clear;
	__m _v, _eps = _m_set1(1e-16f); int n;
	if(src == dst) {
		for(n = 0; n < len && !_m_aligned(dst + n); n++)
			dst[n] = dst[n] < 1e-16f ? 0.f : 1.f / sqrtf(dst[n]);
		if(n) dst += n, len -= n;
		for(n = _m_nsse(len); n--; dst += _m_inc) {
			_v = _m_load(dst);
			_m_store(dst, _m_and(_m_rsqrt(_v), _m_lt(_eps, _v)));
		}
		src = dst;
	}
	else {
		bool isAligned = _m_aligned(size_t(src) | size_t(dst));
		if(isAligned) for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc) {
			_v = _m_load(src);
			_m_store(dst, _m_and(_m_rsqrt(_v), _m_lt(_eps, _v)));
		}
		else for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc) {
			_v = _m_loadu(src);
			_m_storeu(dst, _m_and(_m_rsqrt(_v), _m_lt(_eps, _v)));
		}
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = src[n] < 1e-16f ? 0.f : 1.f / sqrtf(src[n]);
	////! Implementation 2. accurate! will produce NAN if there is negative in src!
	//yuSqrt(src, dst, len);
	//yuInv(dst, dst, len);
}
//==============================================================================P1
///dst = min(src1, src2);
void  yuMin(const float *src1, const float *src2, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_store(dst, _m_min(_m_load(src1), _m_load(src2)));
	else for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_storeu(dst, _m_min(_m_loadu(src1), _m_loadu(src2)));
	for(n = 0; n < res; n++)
		dst[n] = yumin(src1[n], src2[n]);
}
//dst = max(src1, src2);
void  yuMax(const float *src1, const float *src2, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	if(isAligned) for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_store(dst, _m_max(_m_load(src1), _m_load(src2)));
	else for(; n--; src1 += _m_inc, src2 += _m_inc, dst += _m_inc)
		_m_storeu(dst, _m_max(_m_loadu(src1), _m_loadu(src2)));
	for(n = 0; n < res; n++)
		dst[n] = yumax(src1[n], src2[n]);
}
//dst = min(src, s);
void  yuMin(const float *src, float s, float *dst, int len)
{	
	_m_clear;
	__m _s = _m_set1(s); int n;
	if(src == dst) {
		for(n = 0; n < len && !_m_aligned(dst + n); n++)
			if(s < dst[n]) dst[n] = s;
		if(n) dst += n, len -= n;
		for(n = _m_nsse(len); n--; dst += _m_inc)
			_m_store(dst, _m_min(_m_load(dst), _s));
		src = dst;
	}
	else {
		bool isAligned = _m_aligned(size_t(src) | size_t(dst));
		if(isAligned) for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_store(dst, _m_min(_m_load(src), _s));
		else for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_storeu(dst, _m_min(_m_loadu(src), _s));
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = yumin(src[n], s);
}
//dst = max(src, s);
void  yuMax(const float *src, float s, float *dst, int len)
{
	_m_clear;
	__m _s = _m_set1(s); int n;
	if(src == dst) {
		for(n = 0; n < len && !_m_aligned(dst + n); n++)
			if(dst[n] < s) dst[n] = s;
		if(n) dst += n, len -= n;
		for(n = _m_nsse(len); n--; dst += _m_inc)
			_m_store(dst, _m_max(_m_load(dst), _s));
		src = dst;
	}
	else {
		bool isAligned = _m_aligned(size_t(src) | size_t(dst));
		if(isAligned) for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_store(dst, _m_max(_m_load(src), _s));
		else for(n = _m_nsse(len); n--; src += _m_inc, dst += _m_inc)
			_m_storeu(dst, _m_max(_m_loadu(src), _s));
	}
	for(len = _m_res(len), n = 0; n < len; n++)
		dst[n] = yumax(src[n], s);
}
//==============================================================================P1
///dst = src1 == src2;
void  yuCompareEq(const char *src1, const char *src2, uchar *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	__mi *_src1 = (__mi*)src1, *_src2 = (__mi*)src2, *_dst = (__mi*)dst;
#ifdef USE_AVX
	int n = len >> 5, res = len & 31; //n = len / 32, res = len % 32;
	if(isAligned) while(n--)
		_mi_store(_dst++, _mm256_cmpeq_epi8(_mi_load(_src1++), _mi_load(_src2++)));
	else while(n--)
		_mi_storeu(_dst++, _mm256_cmpeq_epi8(_mi_loadu(_src1++), _mi_loadu(_src2++)));
#else
	int n = len >> 4, res = len & 15; //int n = len / 16, res = len % 16;
	if(isAligned) while(n--)
		_mi_store(_dst++, _mm_cmpeq_epi8(_mi_load(_src1++), _mi_load(_src2++)));
	else while(n--)
		_mi_storeu(_dst++, _mm_cmpeq_epi8(_mi_loadu(_src1++), _mi_loadu(_src2++)));
#endif
	if(res) {
		src1 = (char*)_src1, src2 = (char*)_src2, dst = (uchar*)_dst;
		for(n = 0; n < res; n++)
			dst[n] = src1[n] == src2[n] ? 0xff : 0;
	}
}
///dst = src == s;
void  yuCompareEq(const char *src, char s, uchar *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src) | size_t(dst));
	__mi *_src = (__mi*)src, *_dst = (__mi*)dst;
#ifdef USE_AVX
	int n = len >> 5, res = len & 31; //int n = len / 32, res = len % 32;
	__mi _s = _mm256_set1_epi8(s);
	if(isAligned) while(n--)
		_mi_store(_dst++, _mm256_cmpeq_epi8(_mi_load(_src++), _s));
	else while(n--)
		_mi_storeu(_dst++, _mm256_cmpeq_epi8(_mi_loadu(_src++), _s));
#else
	int n = len >> 4, res = len & 15; //int n = len / 16, res = len % 16;
	__mi _s = _mm_set1_epi8(s);
	if(isAligned) while(n--)
		_mi_store(_dst++, _mm_cmpeq_epi8(_mi_load(_src++), _s));
	else while(n--)
		_mi_storeu(_dst++, _mm_cmpeq_epi8(_mi_loadu(_src++), _s));
#endif
	if(res) {
		src = (char*)_src, dst = (uchar*)_dst;
		for(n = 0; n < res; n++)
			dst[n] = src[n] == s ? 0xff : 0; //char(-128) is uchar(255)
	}
}
///dst = src1 == src2; applicable to both int32 & float type.
void  yuCompareEq(const int *src1, const int *src2, uchar *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2));
	int n = _m_nsse(len), res = _m_res(len);
	__mi *_src1 = (__mi*)src1, *_src2 = (__mi*)src2, *_buf = (__mi*)_align_buf;
	uchar *buf = (uchar*)_buf;
#define CMD1 dst[0] = buf[0]; dst[1] = buf[4]; dst[2] = buf[8]; dst[3] = buf[12]
#ifdef USE_AVX
#define CMD CMD1; dst[4] = buf[16]; dst[5] = buf[20]; dst[6] = buf[24]; dst[7] = buf[28]
#else
#define CMD CMD1
#endif
	if(isAligned) for(; n--; dst += _m_inc) {
		_mi_store(_buf, _mi_eq(_mi_load(_src1++), _mi_load(_src2++)));
		CMD;
	}
	else for(; n--; dst += _m_inc) {
		_mi_storeu(_buf, _mi_eq(_mi_loadu(_src1++), _mi_loadu(_src2++)));
		CMD;
	}
	src1 = (int*)_src1, src2 = (int*)_src2;
	for(n = 0; n < res; n++)
		dst[n] = src1[n] == src2[n] ? 0xff : 0;
}
///dst = src == s; applicable to both int32 & float type.
void  yuCompareEq(const int *src, int s, uchar *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(src);
	int n = _m_nsse(len), res = _m_res(len);
	__mi *_src = (__mi*)src, *_buf = (__mi*)_align_buf, _s = _mi_set1(s);
	uchar *buf = (uchar*)_buf;
	if(isAligned) for(; n--; dst += _m_inc) {
		_mi_store(_buf, _mi_eq(_mi_load(_src++), _s));
		CMD;
	}
	else for(; n--; dst += _m_inc) {
		_mi_store(_buf, _mi_eq(_mi_loadu(_src++), _s));
		CMD;
	}
	src = (int*)_src;
	for(n = 0; n < res; n++)
		dst[n] = src[n] == s ? 0xff : 0;
}
//dst = src < s
void  yuCompareLt(const float *src, float s, uchar *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(src);
	int n = _m_nsse(len), res = _m_res(len);
	uchar *buf = (uchar*)_align_buf;
	__m _s = _m_set1(s);
	if(isAligned) for(; n--; src += _m_inc, dst += _m_inc) {
		_m_store(_align_buf, _m_lt(_m_load(src), _s));
		CMD;
	}
	else for(; n--; src += _m_inc, dst += _m_inc) {
		_m_store(_align_buf, _m_lt(_m_loadu(src), _s));
		CMD;
	}
	for(n = 0; n < res; n++)
		dst[n] = src[n] < s ? 0xff : 0;
}
//dst = src > s
void  yuCompareGt(const float *src, float s, uchar *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(src);
	int n = _m_nsse(len), res = _m_res(len);
	uchar *buf = (uchar*)_align_buf;
	__m _s = _m_set1(s);
	if(isAligned) for(; n--; src += _m_inc, dst += _m_inc) {
		_m_store(_align_buf, _m_lt(_s, _m_load(src)));
		CMD;
	}
	else for(; n--; src += _m_inc, dst += _m_inc) {
		_m_store(_align_buf, _m_lt(_s, _m_loadu(src)));
		CMD;
	}
	for(n = 0; n < res; n++)
		dst[n] = s < src[n] ? 0xff : 0;
}
//dst = src > s, integer version
void  yuCompareGt(const int *src, int s, uchar *dst, int len)
{
	_m_clear;
	__mi *ps = (__mi*)src, *pd = (__mi*)_align_buf, _s = _mi_set1(s);
	bool isAligned = _m_aligned(src);
	int n = _m_nsse(len), res = _m_res(len);
	uchar *buf = (uchar*)_align_buf;
	if(isAligned) for(; n--; dst += _m_inc) {
		_mi_store(pd, _mi_gt(_mi_load(ps++), _s));
		CMD;
	}
	else for(; n--; dst += _m_inc) {
		_mi_store(pd, _mi_gt(_mi_loadu(ps++), _s));
		CMD;
	}
	for(n = 0; n < res; n++)
		dst[n] = s < src[n] ? 0xff : 0;
#undef CMD1
#undef CMD
}
//==============================================================================P1
///dst = src1 & src2; applicable for any type of data.
void  yuAnd(const char *src1, const char *src2, char *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	__mi *_src1 = (__mi*)src1, *_src2 = (__mi*)src2, *_dst = (__mi*)dst;
	int n = len / (_m_inc * 4), res = len % (_m_inc * 4);
	if(isAligned) while(n--)
		_mi_store(_dst++, _mi_and(_mi_load(_src1++), _mi_load(_src2++)));
	else while(n--)
		_mi_storeu(_dst++, _mi_and(_mi_loadu(_src1++), _mi_loadu(_src2++)));
	if(res) {
		src1 = (char*)_src1, src2 = (char*)_src2, dst = (char*)_dst;
		for(n = 0; n < res; n++)
			dst[n] = src1[n] & src2[n];
	}
}
///dst = src1 | src2; applicable for any type of data.
void  yuOr(const char *src1, const char *src2, char *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	__mi *_src1 = (__mi*)src1, *_src2 = (__mi*)src2, *_dst = (__mi*)dst;
	int n = len / (_m_inc * 4), res = len % (_m_inc * 4);
	if(isAligned) while(n--)
		_mi_store(_dst++, _mi_or(_mi_load(_src1++), _mi_load(_src2++)));
	else while(n--)
		_mi_storeu(_dst++, _mi_or(_mi_loadu(_src1++), _mi_loadu(_src2++)));
	if(res) {
		src1 = (char*)_src1, src2 = (char*)_src2, dst = (char*)_dst;
		for(n = 0; n < res; n++)
			dst[n] = src1[n] | src2[n];
	}
}
///dst = (~src1) & src2; applicable for any type of data.
void  yuAndnot(const char *src1, const char *src2, char *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	__mi *_src1 = (__mi*)src1, *_src2 = (__mi*)src2, *_dst = (__mi*)dst;
	int n = len / (_m_inc * 4), res = len % (_m_inc * 4);
	if(isAligned) while(n--)
		_mi_store(_dst++, _mi_andnot(_mi_load(_src1++), _mi_load(_src2++)));
	else while(n--)
		_mi_storeu(_dst++, _mi_andnot(_mi_loadu(_src1++), _mi_loadu(_src2++)));
	if(res) {
		src1 = (char*)_src1, src2 = (char*)_src2, dst = (char*)_dst;
		for(n = 0; n < res; n++)
			dst[n] = (~src1[n]) & src2[n];
	}
}
///dst = src & s; for other data type, see yuNeg() for reference.
void  yuAnd(const char *src, char s, char *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src) | size_t(dst));
	__mi *_src = (__mi*)src, *_dst = (__mi*)dst;
	int n = len / (_m_inc * 4), res = len % (_m_inc * 4);
#ifdef USE_AVX
	__mi _s = _mm256_set1_epi8(s);
#else
	__mi _s = _mm_set1_epi8(s);
#endif
	if(isAligned) while(n--)
		_mi_store(_dst++, _mi_and(_mi_load(_src++), _s));
	else while(n--)
		_mi_storeu(_dst++, _mi_and(_mi_loadu(_src++), _s));
	if(res) {
		src = (char*)_src, dst = (char*)_dst;
		for(n = 0; n < res; n++)
			dst[n] = src[n] & s;
	}
}
///dst = src | s;
void  yuOr(const char *src, char s, char *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src) | size_t(dst));
	__mi *_src = (__mi*)src, *_dst = (__mi*)dst;
	int n = len / (_m_inc * 4), res = len % (_m_inc * 4);
#ifdef USE_AVX
	__mi _s = _mm256_set1_epi8(s);
#else
	__mi _s = _mm_set1_epi8(s);
#endif
	if(isAligned) while(n--)
		_mi_store(_dst++, _mi_or(_mi_load(_src++), _s));
	else while(n--)
		_mi_storeu(_dst++, _mi_or(_mi_loadu(_src++), _s));
	if(res) {
		src = (char*)_src, dst = (char*)_dst;
		for(n = 0; n < res; n++)
			dst[n] = src[n] | s;
	}
}
///dst = (~src) & s;
void  yuAndnot(const char *src, char s, char *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src) | size_t(dst));
	__mi *_src = (__mi*)src, *_dst = (__mi*)dst;
	int n = len / (_m_inc * 4), res = len % (_m_inc * 4);
#ifdef USE_AVX
	__mi _s = _mm256_set1_epi8(s);
#else
	__mi _s = _mm_set1_epi8(s);
#endif
	if(isAligned) while(n--)
		_mi_store(_dst++, _mi_andnot(_mi_load(_src++), _s));
	else while(n--)
		_mi_storeu(_dst++, _mi_andnot(_mi_loadu(_src++), _s));
	if(res) {
		src = (char*)_src, dst = (char*)_dst;
		for(n = 0; n < res; n++)
			dst[n] = (~src[n]) & s;
	}
}
//==============================================================================P1
//dst = (src1.*src2) >> numBits
//It's easily seen that only _mm_mul_epi32 (rather than _mm_mul_epu32) can produce correct results when applying shifting.
void  yuMulShift(const int *src1, const int *src2, int *dst, int len, int numBits)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__mi *_src1 = (__mi*)src1, *_src2 = (__mi*)src2, *_dst = (__mi*)dst, a, b;
	if(isAligned) while(n--) {
		a = _mi_load(_src1++), b = _mi_load(_src2++);
		_mi_store(_dst++, _mi_6432(_mi_sr64(_mi_mul64(a, b), numBits), _mi_sr64(_mi_mul64p(a, b), numBits)));
	}
	else while(n--) {
		a = _mi_loadu(_src1++), b = _mi_load(_src2++);
		_mi_storeu(_dst++, _mi_6432(_mi_sr64(_mi_mul64(a, b), numBits), _mi_sr64(_mi_mul64p(a, b), numBits)));
	}
	if(res) {
		src1 = (int*)_src1, src2 = (int*)_src2, dst = (int*)_dst;
		for(n = 0; n < res; n++)
			dst[n] = ((int64)src1[n] * (int64)src2[n]) >> numBits;
	}
}
///dst = (src1.*src2 - src3^2) >> numBits;
void  yuDetShift(const int *src1, const int *src2, const int *src3, int *dst, int len, int numBits)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(src3) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__mi *ps1 = (__mi*)src1, *ps2 = (__mi*)src2, *ps3 = (__mi*)src3, *pd = (__mi*)dst, a, b, c, d;
	if(isAligned) while(n--) {
		a = _mi_load(ps1++), b = _mi_load(ps2++), c = _mi_load(ps3++), d = _mi_sr(c, 4);
		c = _mi_sub(_mi_mul64(a, b), _mi_mul64(c, c));
		d = _mi_sub(_mi_mul64p(a, b), _mi_mul64(d, d));
		_mi_store(pd++, _mi_6432(_mi_sr64(c, numBits), _mi_sr64(d, numBits)));
	}
	else while(n--) {
		a = _mi_loadu(ps1++), b = _mi_loadu(ps2++), c = _mi_loadu(ps3++), d = _mi_sr(c, 4);
		c = _mi_sub(_mi_mul64(a, b), _mi_mul64(c, c));
		d = _mi_sub(_mi_mul64p(a, b), _mi_mul64(d, d));
		_mi_storeu(pd++, _mi_6432(_mi_sr64(c, numBits), _mi_sr64(d, numBits)));
	}
	if(res) {
		src1 = (int*)ps1, src2 = (int*)ps2, src3 = (int*)ps3, dst = (int*)pd;
		for(n = 0; n < res; n++)
			dst[n] = (int)(((int64)src1[n] * (int64)src2[n] - yusquare((int64)src3[n])) >> numBits);
	}
}
///dst = (src1*src2 - src3*src4) >> numBits;
void  yuMulSubShift(const int *src1, const int *src2, const int *src3, const int *src4, int *dst, int len, int numBits)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(src3) | size_t(src4) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__mi *ps1 = (__mi*)src1, *ps2 = (__mi*)src2, *ps3 = (__mi*)src3, *ps4 = (__mi*)src4, *pd = (__mi*)dst, a, b, c, d;
	if(isAligned) while(n--) {
		a = _mi_load(ps1++), b = _mi_load(ps2++), c = _mi_load(ps3++), d = _mi_load(ps4++);
		c = _mi_sub(_mi_mul64(a, b), _mi_mul64(c, d));
		d = _mi_sub(_mi_mul64p(a, b), _mi_mul64p(c, d));
		_mi_store(pd++, _mi_6432(_mi_sr64(c, numBits), _mi_sr64(d, numBits)));
	}
	else while(n--) {
		a = _mi_loadu(ps1++), b = _mi_loadu(ps2++), c = _mi_loadu(ps3++), d = _mi_loadu(ps4++);
		c = _mi_sub(_mi_mul64(a, b), _mi_mul64(c, d));
		d = _mi_sub(_mi_mul64p(a, b), _mi_mul64p(c, d));
		_mi_storeu(pd++, _mi_6432(_mi_sr64(c, numBits), _mi_sr64(d, numBits)));
	}
	if(res) {
		src1 = (int*)ps1, src2 = (int*)ps2, src3 = (int*)ps3, src4 = (int*)ps4, dst = (int*)pd;
		for(n = 0; n < res; n++)
			dst[n] = (int)(((int64)src1[n] * (int64)src2[n] - (int64)src3[n] * (int64)src4[n]) >> numBits);
	}
}
//==============================================================================P1
//Define a function object with member function "operator()" to replace the original comparison operation.
template<class T>
class compare_func {
public:
	const T *A;
	compare_func(const T *arr) : A(arr) {}
	bool  operator() (uchar &Left, uchar &Right) {
		return A[Left] < A[Right]; // sort in increasing order
	}
	bool  operator() (int &Left, int &Right) {
		return A[Left] < A[Right];
	}
	bool  operator() (unsigned &Left, unsigned &Right) {
		return A[Left] < A[Right];
	}
};
//Sort an array in increasing order, just get the indexes
template<class T1, class T2>
inline void  _yuSortIdx(const T1 *arr, T2 *idx, T2 len)
{
	T2 *idx2 = idx;
	for(T2 i = 0; i < len;)
		*idx2++ = i++;
	std::sort(idx, idx2, compare_func<T1>(arr));
}
void  yuSortIdx(const ushort *arr, int *idx, int len)
{
	_yuSortIdx(arr, idx, len);
}
void  yuSortIdx(const int *arr, int *idx, int len)
{
	_yuSortIdx(arr, idx, len);
}
void  yuSortIdx(const float *arr, int *idx, int len)
{
	_yuSortIdx(arr, idx, len);
}
void  yuSortIdx(const float *arr, unsigned *idx, unsigned len)
{
	_yuSortIdx(arr, idx, len);
}

//Sort an array in ascending order and just get the indexes.
const float *yuSortGlobal = 0;
static inline int ascending(const void *left, const void *right)
{
	return yuSortGlobal[*(int*)left] < yuSortGlobal[*(int*)right] ? -1 : (yuSortGlobal[*(int*)right] < yuSortGlobal[*(int*)left] ? 1 : 0);
}
static inline int descending(const void *left, const void *right)
{
	return yuSortGlobal[*(int*)left] < yuSortGlobal[*(int*)right] ? 1 : (yuSortGlobal[*(int*)left] == yuSortGlobal[*(int*)right] ? 0 : -1);
}
//pure C; slower than yuSortIdx()
void yuSort(const float *arr, int *idx, int len)
{
	for(int k = 0; k < len; k++)
		idx[k] = k;
	yuSortGlobal = arr;
	qsort(idx, len, 4, ascending);
}

static inline int ascendingU32(const void *left, const void *right)
{
	return yuSortGlobal[*(unsigned*)left] < yuSortGlobal[*(unsigned*)right] ? -1 : (yuSortGlobal[*(unsigned*)left] == yuSortGlobal[*(unsigned*)right] ? 0 : 1);
}
void yuSort(const float *arr, unsigned *idx, unsigned len)
{
	for(unsigned k = 0; k < len; k++)
		idx[k] = k;
	yuSortGlobal = arr;
	qsort(idx, len, 4, ascendingU32);
}









//==============================================================================P2
//ret = sum(src);
inline double _yuSum(const float *src, int len)
{
	assert(_m_aligned(src));
	__m _s = _m_zero(); int n;
	for(n = _m_nsse(len); n--; src += _m_inc)
		_s = _m_add(_s, _m_load(src));
	_m_store(_align_buf, _s);
	double s = _m_sumd(_align_buf);
	for(len = _m_res(len), n = 0; n < len; n++)
		s += src[n];
	return s;
}
///ret = dot(src, src) = sum(src.^2);
inline double _yuDot(const float *src, int len)
{
	assert(_m_aligned(src));
	__m _s = _m_zero(), _v; int n;
	for(n = _m_nsse(len); n--; src += _m_inc) {
		_v = _m_load(src);
		_s = _m_add(_s, _m_mul(_v, _v));
	}
	_m_store(_align_buf, _s);
	double s = _m_sumd(_align_buf);
	for(len = _m_res(len), n = 0; n < len; n++)
		s += src[n] * src[n];
	return s;
}
///ret = dot(src1, src2) = sum(src1 .* src2);
inline double _yuDot(const float *src1, const float *src2, int len)
{
	__m _s = _m_zero(); int n;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2));
	if(isAligned) for(n = _m_nsse(len); n--; src1 += _m_inc, src2 += _m_inc)
		_s = _m_add(_s, _m_mul(_m_load(src1), _m_load(src2)));
	else for(n = _m_nsse(len); n--; src1 += _m_inc, src2 += _m_inc)
		_s = _m_add(_s, _m_mul(_m_loadu(src1), _m_loadu(src2)));
	_m_store(_align_buf, _s);
	double s = _m_sumd(_align_buf);
	for(len = _m_res(len), n = 0; n < len; n++)
		s += src1[n] * src2[n];
	return s;
}
///ret = dot(src1, src2, src3) = sum(src1 .* src2 .* src3);
inline double _yuDot(const float *src1, const float *src2, const float *src3, int len)
{
	__m _s = _m_zero(); int n;
	bool isAligned = _m_aligned(size_t(src1) | size_t(src2) | size_t(src3));
	if(isAligned) for(n = _m_nsse(len); n--; src1 += _m_inc, src2 += _m_inc, src3 += _m_inc)
		_s = _m_add(_s, _m_mul(_m_mul(_m_load(src1), _m_load(src2)), _m_load(src3)));
	else for(n = _m_nsse(len); n--; src1 += _m_inc, src2 += _m_inc, src3 += _m_inc)
		_s = _m_add(_s, _m_mul(_m_mul(_m_loadu(src1), _m_loadu(src2)), _m_loadu(src3)));
	_m_store(_align_buf, _s);
	double s = _m_sumd(_align_buf);
	for(len = _m_res(len), n = 0; n < len; n++)
		s += src1[n] * src2[n] * src3[n];
	return s;
}
double  yuSum(const float *src, int len)
{
	_m_clear;
	double s = 0.0; int n = 0;
	for(n = 0; n < len && !_m_aligned(src); n++)
		s += *src++;
	for(; n + _batch_sz < len; n += _batch_sz, src += _batch_sz)
		s += _yuSum(src, _batch_sz);
	if(n < len)
		s += _yuSum(src, len - n);
	return s;
}
double  yuDot(const float *src, int len)
{
	_m_clear;
	double s = 0.0; int n = 0;
	for(n = 0; n < len && !_m_aligned(src + n); n++)
		s += src[n] * src[n];
	if(n) src += n, len -= n;
	for(n = 0; n + _batch_sz < len; n += _batch_sz, src += _batch_sz)
		s += _yuDot(src, _batch_sz);
	if(n < len)
		s += _yuDot(src, len - n);
	return s;
}
double  yuDot(const float *src1, const float *src2, int len)
{
	if(src1 == src2)
		return yuDot(src1, len);
	_m_clear;
	double s = 0.0; int n = 0;
	for(; n + _batch_sz < len; n += _batch_sz, src1 += _batch_sz, src2 += _batch_sz)
		s += _yuDot(src1, src2, _batch_sz);
	if(n < len)
		s += _yuDot(src1, src2, len - n);
	return s;
}
double  yuDot(const float *src1, const float *src2, const float *src3, int len)
{
	_m_clear;
	double s = 0.0; int n = 0;
	for(; n + _batch_sz < len; n += _batch_sz, src1 += _batch_sz, src2 += _batch_sz, src3 += _batch_sz)
		s += _yuDot(src1, src2, src3, _batch_sz);
	if(n < len)
		s += _yuDot(src1, src2, src3, len - n);
	return s;
}
double  yuSum(const double *src, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(src);
	double *pbuf = (double*)_align_buf;
#ifdef USE_AVX
	int n = len >> 2, res = len & 3; //int n = len / 4, res = len % 4;
	__m256d _s = _mm256_setzero_pd();
	if(isAligned) for(; n--; src += 4)
		_s = _mm256_add_pd(_s, _mm256_load_pd(src));
	else for(; n--; src += 4)
		_s = _mm256_add_pd(_s, _mm256_loadu_pd(src));
	_mm256_store_pd(pbuf, _s);
	double s = pbuf[0] + pbuf[1] + pbuf[2] + pbuf[3];
#else
	int n = len / 2, res = len % 2;
	__m128d _s = _mm_setzero_pd();
	if(isAligned) for(; n--; src += 2)
		_s = _mm_add_pd(_s, _mm_load_pd(src));
	else for(; n--; src += 2)
		_s = _mm_add_pd(_s, _mm_loadu_pd(src));
	_mm_store_pd(pbuf, _s);
	double s = pbuf[0] + pbuf[1];
#endif
	for(n = 0; n < res; n++)
		s += src[n];
	return s;
}
//There is a nnz() function in mat_basic.cpp; but here we want to specialize for uchar array.
int  yuNNZ(uchar *src, int len)
{
	int n = len >> 5, res = len & 31, cnt = 0;
	for(; n--; src += 32) {
		if(src[0]) cnt++; if(src[1]) cnt++; if(src[2]) cnt++; if(src[3]) cnt++;
		if(src[4]) cnt++; if(src[5]) cnt++; if(src[6]) cnt++; if(src[7]) cnt++;
		if(src[8]) cnt++; if(src[9]) cnt++; if(src[10]) cnt++; if(src[11]) cnt++;
		if(src[12]) cnt++; if(src[13]) cnt++; if(src[14]) cnt++; if(src[15]) cnt++;
		if(src[16]) cnt++; if(src[17]) cnt++; if(src[18]) cnt++; if(src[19]) cnt++;
		if(src[20]) cnt++; if(src[21]) cnt++; if(src[22]) cnt++; if(src[23]) cnt++;
		if(src[24]) cnt++; if(src[25]) cnt++; if(src[26]) cnt++; if(src[27]) cnt++;
		if(src[28]) cnt++; if(src[29]) cnt++; if(src[30]) cnt++; if(src[31]) cnt++;
	}
	for(n = 0; n < res; n++)
		if(src[n])
			cnt++;
	return cnt;
}









//==============================================================================P3
//dst = src';
//src is 2D matrix of size h(rows)-by-w(cols), its row step is sstep;
//dst is 2D matrix of size w(rows)-by-h(cols), its row step is dstep;
void  yuTranspose(const float *src, float *dst, int h, int w, int64 sstep, int64 dstep)
{
	_m_clear;
	int n = _m_nsse(h), res = _m_res(h), k;

	const float *ps[_m_inc] = { src };
	for(k = 1; k < _m_inc; k++)
		ps[k] = ps[k - 1] + sstep;

	Buffer buf; buf.reserve(sizeof(float*) * w);
	float **pd = (float**)buf.p;
	pd[0] = dst;
	for(k = 1; k < w; k++)
		pd[k] = pd[k - 1] + dstep;

#ifdef USE_AVX
	while(n--) {
		for(k = 0; k < w; pd[k++] += _m_inc)
			_m_storeu(pd[k], _mm256_set_ps(ps[7][k], ps[6][k], ps[5][k], ps[4][k], ps[3][k], ps[2][k], ps[1][k], ps[0][k]));
		ps[0] = ps[7] + sstep;
		ps[1] = ps[0] + sstep;
		ps[2] = ps[1] + sstep;
		ps[3] = ps[2] + sstep;
		ps[4] = ps[3] + sstep;
		ps[5] = ps[4] + sstep;
		ps[6] = ps[5] + sstep;
		ps[7] = ps[6] + sstep;
	}
#else
	while(n--) {
		for(k = 0; k < w; pd[k++] += _m_inc)
			_m_storeu(pd[k], _mm_set_ps(ps[3][k], ps[2][k], ps[1][k], ps[0][k]));
		ps[0] = ps[3] + sstep;
		ps[1] = ps[0] + sstep;
		ps[2] = ps[1] + sstep;
		ps[3] = ps[2] + sstep;
	}
#endif
	for(n = 0; n < res; n++) {
		for(k = 0; k < w; k++)
			pd[k][n] = ps[n][k];
	}
}
//==============================================================================P3
//dst = src'; uchar version.
void  yuTranspose(const uchar *src, uchar *dst, int h, int w, int64 sstep, int64 dstep)
{
	_m_clear;
	const int inc = _m_inc * 4;
	int n = h / inc, res = h % inc, k;

	const uchar *ps[inc] = { src };
	for(k = 1; k < inc; k++)
		ps[k] = ps[k - 1] + sstep;

	Buffer buf; buf.reserve(sizeof(uchar*) * w);
	uchar **pd = (uchar**)buf.p;
	pd[0] = dst;
	for(k = 1; k < w; k++)
		pd[k] = pd[k - 1] + dstep;

#ifdef USE_AVX
	while(n--) {
		for(k = 0; k < w; pd[k++] += inc)
			_mi_storeu((__mi*)pd[k], _mm256_set_epi8(
				ps[31][k], ps[30][k], ps[29][k], ps[28][k], ps[27][k], ps[26][k], ps[25][k], ps[24][k],
				ps[23][k], ps[22][k], ps[21][k], ps[20][k], ps[19][k], ps[18][k], ps[17][k], ps[16][k],
				ps[15][k], ps[14][k], ps[13][k], ps[12][k], ps[11][k], ps[10][k], ps[9][k], ps[8][k],
				ps[7][k], ps[6][k], ps[5][k], ps[4][k], ps[3][k], ps[2][k], ps[1][k], ps[0][k]));
		ps[0] = ps[31] + sstep;
		for(k = 1; k < 32; k++)
			ps[k] = ps[k - 1] + sstep;
	}
#else
	while(n--) {
		for(k = 0; k < w; pd[k++] += inc)
			_mi_storeu((__mi*)pd[k], _mm_set_epi8(
				ps[15][k], ps[14][k], ps[13][k], ps[12][k], ps[11][k], ps[10][k], ps[9][k], ps[8][k],
				ps[7][k], ps[6][k], ps[5][k], ps[4][k], ps[3][k], ps[2][k], ps[1][k], ps[0][k]));
		ps[0] = ps[15] + sstep;
		for(k = 1; k < 16; k++)
			ps[k] = ps[k - 1] + sstep;
	}
#endif
	for(n = 0; n < res; n++) {
		for(k = 0; k < w; k++)
			pd[k][n] = ps[n][k];
	}
}
//==============================================================================P3
//Super fast method to initialize a continuous memory.
//dst(0) = ... = dst(nVals-1) = *src;
//src points to an variable of size "valSz"; dst points to an array of "nVals*valSz".
void  yuSetTo(uchar *dst, const uchar *src, size_t valSz, size_t nVals)
{
	if(nVals < 1) {
		//fprintf(stderr, "yuSetTo(): nVals < 1");
		//fflush(stderr), throw;
		return;
	}
	memcpy(dst, src, valSz);
	if(nVals == 1)
		return;
	uchar *dst2 = dst + valSz;
	size_t cpySz = valSz, totSz = valSz * nVals;
	while((cpySz << 1) <= totSz) {
		memcpy(dst2, dst, cpySz);
		dst2 += cpySz;
		cpySz <<= 1;
	}
	if(cpySz < totSz)
		memcpy(dst2, dst, totSz - cpySz);
}
//==============================================================================P3
//Iteratively initialize a super large (exceeding UINT32_MAX) memory.
void  yuLSetTo(uchar *dst, const uchar *src, size_t valSz, uint64 nVals)
{
	uint64 memSz = (uint64)valSz * nVals;
	if(memSz <= (uint64)UINT32_MAX) {
		yuSetTo(dst, src, valSz, (size_t)nVals);
		return;
	}
	size_t nVals1 = UINT32_MAX / valSz;
	yuSetTo(dst, src, valSz, nVals1); //init some memory (memSz <= UINT32_MAX)
	memSz = nVals1 * valSz;
	//Now loop copy dst[memSz] to the who memory storage
	uchar *dst2 = dst + memSz;
	uint64 k = (uint64)nVals1; //num. values already set
	while(k + nVals1 <= nVals) {
		memcpy(dst2, dst, memSz);
		dst2 += memSz;
		k += nVals1;
	}
	if(k < nVals)
		memcpy(dst2, dst, (nVals - k) * valSz);
}
//==============================================================================P3
//dst(mask) = val;
//Warning: require mask(mask!=0) to be 255.
void  yuSetMask(uchar *dst, const uchar *mask, uchar val, int len)
{
	//(1) A = (~mask) & dst; //dst(mask) = 0
	//(2) B = mask & val; //mask(mask) = val
	//(3) dst = A | B;
	_m_clear;
	bool isAligned = _m_aligned(size_t(dst) | size_t(mask));
	__mi *_dst = (__mi*)dst, *_mask = (__mi*)mask, _mmask;
#ifdef USE_AVX
	__mi _val = _mm256_set1_epi8(val);
#else
	__mi _val = _mm_set1_epi8(val);
#endif
	int n = len / (_m_inc * 4), res = len % (_m_inc * 4);
	if(isAligned) for(; n--; _dst++) {
		_mmask = _mi_load(_mask++);
		_mi_store(_dst, _mi_or(_mi_andnot(_mmask, _mi_load(_dst)), _mi_and(_mmask, _val)));
	}
	else for(; n--; _dst++) {
		_mmask = _mi_loadu(_mask++);
		_mi_storeu(_dst, _mi_or(_mi_andnot(_mmask, _mi_loadu(_dst)), _mi_and(_mmask, _val)));
	}
	if(res) {
		mask = (uchar*)_mask, dst = (uchar*)_dst;
		for(n = 0; n < res; n++) if(mask[n])
			dst[n] = val;
	}
}










//==============================================================================P4
///int -> float; support in-place operation.
void  yuConvert(const int *src, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__mi *_src = (__mi*)src;
	if(isAligned) for(; n--; dst += _m_inc)
		_m_store(dst, _mi_cvtps(_mi_load(_src++)));
	else for(; n--; dst += _m_inc)
		_m_storeu(dst, _mi_cvtps(_mi_loadu(_src++)));
	if(res) {
		src = (int*)_src;
		for(n = 0; n < res; n++)
			dst[n] = (float)src[n];
	}
}
//==============================================================================P4
//dst = float(src) * scale;
void  yuConvert(const int *src, float scale, float *dst, int len)
{
	if(yuabs(scale - 1.f) < 1e-16f)
		return yuConvert(src, dst, len);
	_m_clear;
	bool isAligned = _m_aligned(size_t(src) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__mi *_src = (__mi*)src; __m _scale = _m_set1(scale);
	if(isAligned) for(; n--; dst += _m_inc)
		_m_store(dst, _m_mul(_mi_cvtps(_mi_load(_src++)), _scale));
	else for(; n--; dst += _m_inc)
		_m_storeu(dst, _m_mul(_mi_cvtps(_mi_loadu(_src++)), _scale));
	if(res) {
		src = (int*)_src;
		for(n = 0; n < res; n++)
			dst[n] = (float)src[n] * scale;
	}
}
//==============================================================================P4
///float -> int: support in-place operation.
///If isTrunc==true, then dst = int(src); else dst = round(src).
void  yuConvert(const float *src, int *dst, int len, bool isTrunc)
{
	_m_clear;
	bool isAligned = _m_aligned(size_t(src) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__mi *_dst = (__mi*)dst;
	if(isTrunc) {
		if(isAligned) for(; n--; src += _m_inc)
			_mi_store(_dst++, _m_cvttepi32(_m_load(src)));
		else for(; n--; src += _m_inc)
			_mi_storeu(_dst++, _m_cvttepi32(_m_loadu(src)));
	}
	else {
		if(isAligned) for(; n--; src += _m_inc)
			_mi_store(_dst++, _m_cvtepi32(_m_load(src)));
		else for(; n--; src += _m_inc)
			_mi_storeu(_dst++, _m_cvtepi32(_m_loadu(src)));
	}
	dst = (int*)_dst;
	for(n = 0; n < res; n++)
		dst[n] = (int)(isTrunc ? src[n] : (src[n] < 0 ? src[n] - 0.5f : src[n] + 0.5f));
}
//==============================================================================P4
//dst = int(src * scale);
void  yuConvert(const float *src, float scale, int *dst, int len, bool isTrunc)
{
	if(yuabs(scale - 1.f) < 1e-16f)
		return yuConvert(src, dst, len, isTrunc);
	_m_clear;
	bool isAligned = _m_aligned(size_t(src) | size_t(dst));
	int n = _m_nsse(len), res = _m_res(len);
	__mi *_dst = (__mi*)dst; __m _scale = _m_set1(scale);
	if(isTrunc) {
		if(isAligned) for(; n--; src += _m_inc)
			_mi_store(_dst++, _m_cvttepi32(_m_mul(_m_load(src), _scale)));
		else for(; n--; src += _m_inc)
			_mi_storeu(_dst++, _m_cvttepi32(_m_mul(_m_loadu(src), _scale)));
	}
	else{
		if(isAligned) for(; n--; src += _m_inc)
			_mi_store(_dst++, _m_cvtepi32(_m_mul(_m_load(src), _scale)));
		else for(; n--; src += _m_inc)
			_mi_storeu(_dst++, _m_cvtepi32(_m_mul(_m_loadu(src), _scale)));
	}	
	dst = (int*)_dst;
	for(n = 0; n < res; n++) {
		float tmp = src[n] * scale;
		dst[n] = (int)(isTrunc ? tmp : (tmp < 0 ? tmp - 0.5f : tmp + 0.5f));
	}
}
//==============================================================================P4
///uchar -> float;
void  yuConvert(const uchar *src, float *dst, int len)
{
	_m_clear;
	bool isAligned = _m_aligned(dst);
	int n = _m_nsse(len), res = _m_res(len);
#ifdef USE_AVX
	if(isAligned) for(; n--; src += _m_inc, dst += _m_inc)
		_m_store(dst, _mm256_set_ps((float)src[7], (float)src[6], (float)src[5], (float)src[4], (float)src[3], (float)src[2], (float)src[1], (float)src[0]));
	else for(; n--; src += _m_inc, dst += _m_inc)
		_m_storeu(dst, _mm256_set_ps((float)src[7], (float)src[6], (float)src[5], (float)src[4], (float)src[3], (float)src[2], (float)src[1], (float)src[0]));
#else
	if(isAligned) for(; n--; src += _m_inc, dst += _m_inc)
		_m_store(dst, _mm_set_ps((float)src[3], (float)src[2], (float)src[1], (float)src[0]));
	else for(; n--; src += _m_inc, dst += _m_inc)
		_m_storeu(dst, _mm_set_ps((float)src[3], (float)src[2], (float)src[1], (float)src[0]));
#endif
	for(n = 0; n < res; n++)
		dst[n] = (float)src[n];
}
//==============================================================================P4
///int -> uchar;
void  yuConvert(const int *src, uchar *dst, int len)
{
	_m_clear;
	__mi *_dst = (__mi*)dst, _ushorts[2];
#ifdef USE_AVX
	int n = len >> 5, res = len & 31; //int n = len / 32, res = len % 32;
	__mi _ints[2];
	for(; n--; src += 32) {
		_ints[0] = _mm256_set_epi32(src[19], src[18], src[17], src[16], src[3], src[2], src[1], src[0]);
		_ints[1] = _mm256_set_epi32(src[23], src[22], src[21], src[20], src[7], src[6], src[5], src[4]);
		_ushorts[0] = _mm256_packus_epi32(_ints[0], _ints[1]); //8 int + 8 int => 16 ushort
		_ints[0] = _mm256_set_epi32(src[27], src[26], src[25], src[24], src[11], src[10], src[9], src[8]);
		_ints[1] = _mm256_set_epi32(src[31], src[30], src[29], src[28], src[15], src[14], src[13], src[12]);
		_ushorts[1] = _mm256_packus_epi32(_ints[0], _ints[1]);
		_mi_storeu(_dst++, _mm256_packus_epi16(_ushorts[0], _ushorts[1]));
	}
#else
	bool isAligned = _m_aligned(size_t(src) | size_t(dst));
	int n = len >> 4, res = len & 15; //int n = len / 16, res = len % 16;
	__mi *_src = (__mi*)src;
	if(isAligned) while(n--) {
		_ushorts[0] = _mm_packus_epi32(_mi_load(_src), _mi_load(_src + 1)); _src += 2; //8 int + 8 int => 16 ushort
		_ushorts[1] = _mm_packus_epi32(_mi_load(_src), _mi_load(_src + 1)); _src += 2;
		_mi_store(_dst++, _mm_packus_epi16(_ushorts[0], _ushorts[1])); //16 ushort + 16 ushort => 32 uchar
	}
	else while(n--) {
		_ushorts[0] = _mm_packus_epi32(_mi_loadu(_src), _mi_loadu(_src + 1)); _src += 2; //8 int + 8 int => 16 ushort
		_ushorts[1] = _mm_packus_epi32(_mi_loadu(_src), _mi_loadu(_src + 1)); _src += 2;
		_mi_storeu(_dst++, _mm_packus_epi16(_ushorts[0], _ushorts[1])); //16 ushort + 16 ushort => 32 uchar
	}
	src = (int*)_src;
#endif
	dst = (uchar*)_dst;
	for(n = 0; n < res; n++)
		dst[n] = (uchar)src[n];
}