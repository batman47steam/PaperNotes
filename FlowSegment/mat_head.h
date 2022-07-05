#pragma once
#include "yustd.h"

#define USE_OPENCV //comment this line to disable opencv interfaces
//#define USE_FFTW //comment this line to disable fftw functions
//#define USE_MEX //comment this line to disable mex functions

#if !defined(INT64_MAX)
#define INT64_MAX 0x7fffffffffffffff
#endif

namespace yu
{
	///----------------------------------------
	/// * Part 1. matrix definition *
	/// Definition of the matrix data.
	///----------------------------------------

	template<typename T>
	struct Point_
	{
		T x, y;
		Point_() { x = y = 0; }
		Point_(T x0, T y0) { x = x0, y = y0; }
		template<typename T2> Point_(Point_<T2> p) { x = (T)p.x, y = (T)p.y; }
		Point_ operator = (Point_ p) { x = p.x, y = p.y; return p; }
		bool operator == (Point_ p) const { return x == p.x && y == p.y; }
		bool operator != (Point_ p) const { return x != p.x || y != p.y; }
	};

	template<typename T>
	struct Size_
	{
		T w, h;
		Size_() { w = h = 0; }
		Size_(T width, T height) { w = width, h = height; }
		template<typename T2> Size_(Size_<T2> p) { w = (T)p.w, h = (T)p.h; }
		Size_ operator = (Size_ p) { w = p.w, h = p.h; return p; }
		bool operator == (Size_ p) const { return w == p.w && h == p.h; }
		bool operator != (Size_ p) const { return w != p.w || h != p.h; }
	};

	template<typename T>
	struct Size3_
	{
		T w, h, d;
		Size3_() { w = h = d = 0; }
		Size3_(T width, T height, T channs) { w = width, h = height, d = channs; }
		template<typename T2> Size3_(Size3_<T2> p) { w = (T)p.w, h = (T)p.h, d = (T)p.d; }
		bool operator == (Size3_ p) const { return w == p.w && h == p.h && d == p.d; }
		bool operator != (Size3_ p) const { return w != p.w || h != p.h || d != p.d; }
	};

	template<typename T>
	struct Rect_
	{
		T x, y, w, h;
		Rect_() { x = y = w = h = 0; }
		Rect_(T x0, T y0, T width, T height) { x = x0, y = y0, w = width, h = height; }
		template<typename T2> Rect_(Rect_<T2> p) { x = (T)p.x, y = (T)p.y, w = (T)p.w, h = (T)p.h; }
		Rect_ operator = (Rect_ p) { x = p.x, y = p.y, w = p.w, h = p.h; return p; }
		bool operator == (Rect_ p) const { return x == p.x && y == p.y && w == p.w && h == p.h; }
		bool operator != (Rect_ p) const { return x != p.x || y != p.y || w != p.w || h != p.h; }
		Rect_ operator & (Rect_ p) const {
			T x0 = yumax(x, p.x), x1 = yumin(x + w, p.x + p.w); if(x1 <= x0) return Rect_();
			T y0 = yumax(y, p.y), y1 = yumin(y + h, p.y + p.h); if(y1 <= y0) return Rect_();
			return Rect_(x0, y0, x1 - x0, y1 - y0);
		}
		Rect_ operator | (Rect_ p) const {
			T x0 = yumin(x, p.x), y0 = yumin(y, p.y), x1 = yumax(x + w, p.x + p.w), y1 = yumax(y + h, p.y + p.h);
			return Rect_(x0, y0, x1 - x0, y1 - y0);
		}
		Point_<T> tl() const { return Point_<T>(x, y); }
		Size_<T> size() const { return Size_<T>(w, h); }
		T  area() { return w * h; }
	};

	typedef Point_<int> Point;
	typedef Point_<double> Pointd;
	typedef Size_<int> Size;
	typedef Size_<double> Sized;
	typedef Size3_<int> Size3;
	typedef Size3_<int64> Size3l;
	typedef Rect_<int> Rect;
	typedef Rect_<double> Rectd;

	enum MatType { INVALID, UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, DOUBLE64 };

	template<typename T>
	inline MatType getMatType() {
		size_t sz = sizeof(T);
		bool isInt = double(T(1.1)) == 1, noSign = double(T(-1)) != -1;
		if(sz == 1) return isInt ? (noSign ? UINT8 : INT8) : INVALID;
		if(sz == 2) return isInt ? (noSign ? UINT16 : INT16) : INVALID;
		if(sz == 4) return isInt ? (noSign ? UINT32 : INT32) : (noSign ? INVALID : FLOAT32);
		if(sz == 8) return isInt ? INVALID : (noSign ? INVALID : DOUBLE64);
		return INVALID;
	}
	
	template<typename T, T Tmax> //T should be int/unsigned/int64/uint64
	class  Mat_
	{
	public:
		MatType type;
		uchar *p0; //initial data address (private)
		uchar *p; //pointer to the data, row-major, accord with OpenCV
		int *refcount; //reference count
		T h, w, d; //rows, cols, num. channels
		T step; //row stride, num. of bytes in each row (no smaller than w*d*elemSz())

		T elemSz() const { //size of each element
			if(type == FLOAT32 || type == INT32 || type == UINT32) return 4;
			if(type == DOUBLE64) return 8;
			if(type == UINT8 || type == INT8) return 1;
			if(type == UINT16 || type == INT16) return 2;
			yuErr("Mat::elemSz(): invalid type!");
		}
		T step1() const { return step / elemSz(); }
		Size_<T> size() const { return Size_<T>(w, h); }
		Size3_<T> size3() const { return Size3_<T>(w, h, d); }
		Rect_<T> rect() const { return Rect_<T>(0, 0, w, h); }

		void  release() {
			if(refcount) (*refcount)--;
			if(p0 && refcount && !(*refcount)) delete[] p0;
			type = INVALID;
			p = p0 = 0; refcount = 0;
			h = w = d = step = 0;
		}
		~Mat_() { release(); }

		void  create(T rows, T cols, T channs, MatType _type) {
			if(h == rows && w == cols && d == channs && type == _type)
				return;
			release();
			type = _type;
			float bytes = (float)rows * (float)cols * (float)channs * (float)elemSz();
			if(bytes > (float)Tmax)
				yuErr("Mat::craete(): required memory size exceeds T_MAX! %g > %g", bytes, (float)Tmax);
			h = rows, w = cols, d = channs, step = w * d * elemSz();
			p0 = new uchar[_align_sz + h * step + sizeof(int)]; //_mm_alloc(), _mm_free()
			if(!p0)
				yuErr("Mat::create(): data allocation error!");
			for(p = p0; p != p0 + _align_sz; p++) if(_m_aligned(p)) break;
			refcount = (int*)(p + h * step); *refcount = 1;
		}

		bool  empty() const { return !p0 || type == INVALID || !h || !w || !d; }
		bool  isContinuous() const { return (h == 1 || step == w * d * elemSz()); }		

		//----- Constructions ----
		Mat_() {
			type = INVALID;
			p = p0 = 0; refcount = 0;
			h = w = d = step = 0;
		}
		Mat_(T rows, T cols, T channs, MatType _type) {
			type = INVALID;
			p = p0 = 0; refcount = 0;
			h = w = d = step = 0;
			create(rows, cols, channs, _type);
		}
		Mat_(T rows, T cols, T channs, MatType _type, void *data) {
			type = _type;
			p = p0 = (uchar*)data; refcount = 0;
			h = rows, w = cols, d = channs, step = w * d * elemSz();
		}
		Mat_(T rows, T cols, T channs, MatType _type, void *data, T _step) {
			type = _type;
			p = p0 = (uchar*)data; refcount = 0;
			h = rows, w = cols, d = channs, step = _step;
			if(step < w * d * elemSz())
				yuErr("Mat::Mat(): step < w * d * elemSz()");
		}
		Mat_(const Mat_ &m) {
			type = m.type;
			p = m.p, p0 = m.p0, refcount = m.refcount;
			h = m.h, w = m.w, d = m.d, step = m.step;
			if(refcount) (*refcount)++;
		}
		void operator = (const Mat_ &m) {
			release();
			type = m.type;
			p = m.p, p0 = m.p0, refcount = m.refcount;
			h = m.h, w = m.w, d = m.d, step = m.step;
			if(refcount) (*refcount)++;
		}
		bool operator == (const Mat_ &m) const { //check if they have same size and type
			return h == m.h && w == m.w && d == w.d && type == w.type;
		}
		bool operator != (const Mat_ &m) const {
			return h != m.h || w != m.w || d != m.d || type != m.type;
		}

		//----- ROI ----
		Mat_ row(T y) const {
			if(y < 0 || y >= h)
				yuErr("Mat::row(): y < 0 || y >= h");
			return Mat_(1, w, d, type, p + y * step);
		}
		Mat_ col(T x) const {
			if(x < 0 || x >= w)
				yuErr("Mat::col(): x < 0 || x >= w");
			return Mat_(h, 1, d, type, p + x * d * elemSz(), step);
		}
		Mat_ rows(T yst, T ynd) const {
			if(yst < 0 || yst >= h || ynd < yst || ynd > h)
				yuErr("Mat::rows(): yst < 0 || yst >= h || ynd < yst || ynd > h");
			if(yst == ynd) return Mat_();
			if(yst + 1 == ynd) return row(yst);
			return Mat_(ynd - yst, w, d, type, p + yst * step, step);
		}
		Mat_ cols(T xst, T xnd) const {
			if(xst < 0 || xst >= w || xnd < xst || xnd > w)
				yuErr("Mat::cols(): xst < 0 || xst >= w || xnd < xst || xnd > w");
			if(xst == xnd) return Mat_();
			return Mat_(h, xnd - xst, d, type, p + xst * d * elemSz(), step);
		}
		Mat_ roi(T x, T y, T width, T height) const {
			if(x < 0 || y < 0 || width < 1 || height < 1 || x + width > w || y + height > h)
				yuErr("Mat::roi(): out of range!");
			if(height == 1)
				return Mat_(1, width, d, type, p + y * step + x * d * elemSz());
			return Mat_(height, width, d, type, p + y * step + x * d * elemSz(), step);
		}
		Mat_ operator() (Rect_<T> rect) const { return roi(rect.x, rect.y, rect.w, rect.h); }

		//----- Initialize the memory ----
		void  copyTo(Mat_ &m) const {
			m.create(h, w, d, type);
			if(isContinuous() && m.isContinuous())
				memcpy(m.p, p, h * step);
			else {
				uchar *src = p, *dst = m.p;
				T _step = w * d * elemSz();
				for(int k = (int)h; k--; src += step, dst += m.step)
					memcpy(dst, src, _step);
			}
		}
		Mat_  clone() const {
			Mat_ m;
			copyTo(m);
			return m;
		}
		void  setUchar(uchar val) { //val in [0,255]
			if(isContinuous())
				memset(p, val, h * step);
			else {
				uchar *dst = p;
				T _step = w * d * elemSz();
				for(int k = (int)h; k--; dst += step)
					memset(dst, val, _step);
			}
		}
		void  setZero() { 
			setUchar(0); }
		static Mat_ zeros(T rows, T cols, T channs, MatType _type) {
			Mat_ m(rows, cols, channs, _type);
			m.setZero();
			return m;
		}
		template<typename T1>
		void  setTo(T1 val) {
			T totalSz = h * w * d * elemSz(); size_t valSz = sizeof(val);
			if(totalSz % (T)valSz)
				yuErr("Mat::setTo(): totalSz not integer times of valSz!");
			T nVals = totalSz / (T)valSz;
			if(isContinuous())
				yuLSetTo(p, (uchar*)&val, valSz, (uint64)nVals);
			else {
				row(0).setTo(val);
				uchar *dst = p + step;
				T _step = w * d * elemSz();
				for(int k = 1; k++ < (int)h; dst += step)
					memcpy(dst, p, _step);				
			}
		}	

		//----- Access a specific element/location ----
		//Warning: the ptr/at functions treat the matrices as 2D.
		template<typename T1> T1* ptr(T y, T x) const {
			if(x < 0 || x >= w*d || y < 0 || y >= h)
				yuErr("Mat::ptr(): x < 0 || x >= w*d || y < 0 || y >= h");
			return (T1*)(p + y * step) + x;
		}
		template<typename T1> T1* ptr(Point_<T> p) const {
			return ptr<T1>(p.y, p.x); }
		template<typename T1> T1* ptr(T y) const {
			if(y < 0 || y >= h)
				yuErr("Mat::ptr(): y < 0 || y >= h");
			return (T1*)(p + y * step);
		}
		template<typename T1> T1& at(Point_<T> p)  const {
			return *ptr<T1>(p.y, p.x); }
		template<typename T1> T1& at(T y, T x)  const { 
			return *ptr<T1>(y, x); }
		template<typename T1> T1& at(T y) const {
			if(y < 0 || y >= w*h*d || !isContinuous())
				yuErr("Mat::at(): y < 0 || y >= w*h*d || !isContinuous()");
			return *((T1*)p + y);
		}
	};

	typedef Mat_<int, INT_MAX> Mat; //memory size is limited to INT_MAX (about 2^31) bytes
	typedef Mat_<int64, INT64_MAX> LargeMat;
	
	enum BorderType { BORDER_NONE, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_SYMMETRIC };

	///---------------------------------------------------
	/// * Part 2. functions *
	/// A list of basic functions.
	///---------------------------------------------------	
	inline int sub2ind(int x, int y, int chnn, const Mat &m) {
		return y * m.w * m.d + x * m.d + chnn;
	}
	//x is the column index, y is the row index, return the channel index
	inline int ind2sub(int idx, int &x, int &y, const Mat &m) {
		y = idx / (m.w * m.d); idx -= y * m.w * m.d; x = idx / m.d; return idx - x * m.d;
	}

	//----------------------- mat_save.cpp -----------------------
	void  save(std::string name, const Mat &m);
	Mat   load(std::string name, bool throwError = true);
	//----------------------- mat_basic.cpp -----------------------
	//----------------------- Part 1. -----------------------
	//dst = src1 [+;-] src2;
	void  add(const Mat &src1, const Mat &src2, Mat &dst);
	void  sub(const Mat &src1, const Mat &src2, Mat &dst);
	//dst = src + beta;
	void  add(const Mat &src, double beta, Mat &dst);
	//dst = beta -src; specialized for beta=0.
	void  sub(double beta, const Mat &src, Mat &dst);
	//dst = abs([src;src1-src2;src-beta]);
	void  abs(const Mat &src, Mat &dst);
	void  absSub(const Mat &src1, const Mat &src2, Mat &dst);
	void  absSub(const Mat &src, double beta, Mat &dst);
	//dst = (src1 + src2) - (src3 + src4);
	void  addSubAdd(const Mat &src1, const Mat &src2, const Mat &src3, const Mat &src4, Mat &dst);
	//dst = src .* [src2;alpha]; specialized for src2=src.
	void  mul(const Mat &src1, const Mat &src2, Mat &dst);
	void  mul(const Mat &src, double alpha, Mat &dst);
	//dst = alpha ./ src; with zero-check; specialized for alpha=1.
	void  div(double alpha, const Mat &src, Mat &dst);
	//dst = src1*alpha [+;-] [src2;beta];
	void  scaleAdd(const Mat &src1, double alpha, const Mat &src2, Mat &dst);
	void  scaleAdd(const Mat &src, double alpha, double beta, Mat &dst);
	void  scaleSub(const Mat &src1, double alpha, const Mat &src2, Mat &dst);
	//dst = (src1 [+;-] src2) * alpha;
	void  addScale(const Mat &src1, const Mat &src2, double alpha, Mat &dst);
	void  subScale(const Mat &src1, const Mat &src2, double alpha, Mat &dst);
	//dst = src1 ./ (src2 + beta); no zero-check.
	void  divAdd(const Mat &src1, const Mat &src2, double beta, Mat &dst);
	//dst = (src1 - src2) * alpha + src3;
	void  subScaleAdd(const Mat &src1, const Mat &src2, double alpha, const Mat &src3, Mat &dst);
	//dst = gamma * src ./ (alpha * src + beta);
	void  scaleDivScaleAdd(const Mat &src, Mat &dst, double alpha, double beta, double gamma);
	//dst = src1 .* src2 - src3 .* src4; specialized for src4==src3.
	void  mulSubMul(const Mat &src1, const Mat &src2, const Mat &src3, const Mat &src4, Mat &dst);
	//dst = src1.^2 + src2.^2;
	void  sqSum(const Mat &src1, const Mat &src2, Mat &dst);
	//dst = sqrt(src1.^2 + src2.^2);
	void  sqrtSqSum(const Mat &src1, const Mat &src2, Mat &dst);
	//dst = [1./] sqrt(src); no positive-check for sqrt; rsqrt is with positive-check.
	void  sqrt(const Mat &src, Mat &dst);
	void  rsqrt(const Mat &src, Mat &dst);
	//dst = [min;max](src1, [src2;s]);
	void  mmin(const Mat &src1, const Mat &src2, Mat &dst);
	void  mmax(const Mat &src1, const Mat &src2, Mat &dst);
	void  mmin(const Mat &src, double s, Mat &dst);
	void  mmax(const Mat &src, double s, Mat &dst);
	//dst = src1 == [src2;s];
	void  compareEq(const Mat &src1, const Mat &src2, Mat &dst);
	void  compareEq(const Mat &src, double s, Mat &dst);
	void  compareLt(const Mat &src, double s, Mat &dst);
	void  compareGt(const Mat &src, double s, Mat &dst);
	//dst = src1 [and;or;andnot] [src2;s]; bitwise operations.
	void  bitAnd(const Mat &src1, const Mat &src2, Mat &dst);
	void  bitOr(const Mat &src1, const Mat &src2, Mat &dst);
	void  bitAndnot(const Mat &src1, const Mat &src2, Mat &dst);
	void  bitAnd(const Mat &src, double s, Mat &dst);
	void  bitOr(const Mat &src, double s, Mat &dst);
	void  bitAndnot(const Mat &src, double s, Mat &dst);
	//----------------------- Part 2. -----------------------
	//sum(src); sum(src1.*src2); sum(src1.*src2.*src3); specialized for sum(src.^2).
	double  sum(const Mat &src);
	double  dot(const Mat &src1, const Mat &src2);
	double  dot(const Mat &src1, const Mat &src2, const Mat &src3);
	//countNonZero(src);
	int  nnz(const Mat &src);
	//[minimum,minidx] = find(src==min(src(:)));
	void  minMax(const Mat &src, double *minimum = 0, double *maximum = 0, int *minidx = 0, int *maxidx = 0);
	void  minMaxLoc(const Mat &src, double *minimum = 0, double *maximum = 0, Point *minloc = 0, Point *maxloc = 0);
	//find(isnan(src),1);
	bool  isInf(const Mat &src, Point *loc);
	//display a matrix on screen or output it to local file (according to stderr)
	void  print(const Mat &m);
	//----------------------- Part 3. -----------------------
	//dst = src';
	void  transpose(const Mat &src, Mat &dst);
	//dst(mask) = val;
	void  setMask(Mat &dst, const Mat &mask, double val);
	//----------------------- Part 4. -----------------------
	//dst = static_cast<dtype>(src);
	//Warning: cast signed negative to unsigned integer will result in unexpected results.
	//         e.g., uchar(-1) = 255.
	void  convert(const Mat &src, Mat &dst, MatType dtype);
	//dst = round(src * scale); specialized for scale = 1.
	void  round(const Mat &src, float scale, Mat &dst);
	//dst = float(src) * scale; specialized for scale = 1.
	void  convert(const Mat &src, float scale, Mat &dst);
};

/* This C++ project's source files are named in a  unified way:
*  (1) class implementation files are named by the class name;
*  (2) cpp files that only define pure C++ functions are named by a description of those functions;
*  (3) cpp files that define functions with a yu::Mat interface are named by a mat_ prefix;
*/