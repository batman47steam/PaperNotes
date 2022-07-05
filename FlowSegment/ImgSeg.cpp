#include "ImgSeg.h"
#include "mat_head.h"
#include "mat_opencv.h"

//===================================================================================
//===================================================================================

void ImgSeg::init(int rows, int cols, const int areaRange[2])
{
	if(rows < 2 || cols < 2) {
		printf("ImgSeg::init() error: rows < 2 || cols < 2");
		throw;
	}
	h = rows, w = cols;
	h1 = h - 1, w1 = w - 1;
	numNodes = h * w;
	// 这里right可能是包含这个pixel的（left right)的
	//4 edges for each pixel: 
	//  (1) current-to-right : rows*(cols-1)
	//  (2) current-to-bottom : cols*(rows-1)
	//  (3) current-to-bottomRight : (rows-1)*(cols-1)
	//  (4) current-to-bottomLeft : (rows-1)*(cols-1)
	numEdges = h * w1 + h1 * w + h1 * w1 * 2;
	//bufSz一次性的为后面要处理的部分分配内存空间
	size_t bufSz = (numNodes + 1) * sizeof(float) //ctable0
		+ numEdges * 2 * sizeof(int) //edgeNode
		+ numNodes * sizeof(int) //root
		+ numNodes * sizeof(ROOTINFO) //rootInfo
		+ yumax((numNodes + 1) * sizeof(float) //ctable
			+ numEdges * sizeof(float) //edgeWeight
			+ numEdges * sizeof(int), //index			
			numNodes * sizeof(int) //labels
			+ numNodes * sizeof(int) //areas
		);
	buffer.reserve(bufSz + _align_sz);

	//The order of each pointer's memory position should not change!
	ctable0 = (float*)buffer.p; //kept constant
	buffer.p += (numNodes + 1) * sizeof(float);
	edgeNode = (int*)buffer.p; //kept constant
	buffer.p += numEdges * 2 * sizeof(int);
	root = (int*)buffer.p;
	buffer.p += numNodes * sizeof(int);
	rootInfo = (ROOTINFO*)buffer.p;
	buffer.p += numNodes * sizeof(ROOTINFO);
	ctable = (float*)buffer.p;
	buffer.p += (numNodes + 1) * sizeof(float);
	edgeWeight = (float*)buffer.getp();
	buffer.p += numEdges * sizeof(float);
	index = (int*)buffer.p; 
	buffer.p += numEdges * sizeof(int);
	assert(buffer.p - buffer.data <= buffer.sz);
	//"labels" & "areas" only appear in the post-processing stage, we will overwrite edgeWeight & index
	labels = (int*)ctable;
	areas = labels + numNodes;
	
	//In the segment stage, we keep ctable & edgeNode to be constant (if numNodes & segC are not changed)!
	ctable0[0] = 0;
	for(int k = 1; k <= numNodes; k++)
		ctable0[k] = 1.f / (float)k;
	edgeNode[0] = numNodes;

	if(areaRange) {
		minArea = areaRange[0];
		maxArea = areaRange[1];
		if(minArea > maxArea)
			std::swap(minArea, maxArea);
	}
	else
		minArea = maxArea = 0;
}

//sstep的作用是什么
int ImgSeg::segment(const float *src, int sstep)
{
	//前面的Init()应该是主要起到一个分配内存空间的作用
	if(!buffer.data) {
		printf("ImgSeg::segment() error: please call init() first!");
		throw;
	}
	//这个sstep应该大概是说一次下去一行
	if(sstep == 0)
		sstep = w;
	if(sstep < w) {
		printf("ImgSeg::segment() error: sstep < w");
		throw;
	}

#define METHOD_FAST //2 equivalent methods are implemented，k都是用在for循环里面的
	int k, y, x, ind;
#ifndef METHOD_FAST
	//Get edge weights
	//p里面有4个元素，每个元素都是一个指针
	float *p[4];
	p[0] = edgeWeight;//p0和edgeWeight是一个指向的
	p[1] = p[0] + h * w1;
	p[2] = p[1] + h1 * w;
	p[3] = p[2] + h1 * w1;
	for(y = 0; y < h; y++, src += sstep) {
		//使用w1=w-1,是因为你错开了一个像素，设置了len=w1,应该是在第一行进行操作
		yuSub(src, src + 1, p[0], w1); p[0] += w1;
		if(y < h1) {

			yuSub(src, src + sstep, p[1], w); p[1] += w;
			yuSub(src, src + sstep + 1, p[2], w1); p[2] += w1;
			yuSub(src + 1, src + sstep, p[3], w1); p[3] += w1;
		}
	}
	//把前面p[0],p[1],p[2],p[3]里面的值全部都取绝对值
	yuAbs(edgeWeight, edgeWeight, numEdges);
	//Get node pairs for each edge
	if(doEN) {
		int *pn = edgeNode;
		//(1) current-to-right
		k = 0; //k is node index of current pixel
		//这种形式的for循环能停住吗，可能是要自己去试下
		//pn里面存放的，其实就是上面的edgeWeight里面每个edge
		//所对应的两个vertex的序号
		for(y = h; y--; k++) for(x = w1; x--; k++) {
			*pn++ = k; *pn++ = k + 1;
		}
		//(2) current-to-bottom
		k = 0;
		for(y = h1; y--;) for(x = w; x--; k++) {
			*pn++ = k; *pn++ = k + w;
		}
		//(3) current-to-bottomRight
		k = 0;
		for(y = h1; y--; k++) for(x = w1; x--; k++) {
			*pn++ = k; *pn++ = k + w + 1;
		}
		//(4) current-to-bottomLeft
		k = 0;
		for(y = h1; y--; k++) for(x = w1; x--; k++) {
			*pn++ = k + 1; *pn++ = k + w;
		}
	}
#else
	//Get edge weights & node pairs for each edge
	float *pw = edgeWeight;
	const int sse = _m_nsse(w1), res = _m_res(w1), nsse = w1 - res;
	const float *ps = src;
	//先不关注这些，这个应该是把内存对齐之类的
	const bool srcAligned = _m_aligned(size_t(ps) | size_t(sstep * sizeof(float)));
	assert(_m_aligned(edgeWeight)); __m _a, _b, _c, _d;
	for(y = h1; y--; ps += sstep - nsse) {
		if(srcAligned) for(k = sse; k--; ps += _m_inc) {
			_a = _m_load(ps), _b = _m_loadu(ps + 1), _c = _m_load(ps + sstep), _d = _m_loadu(ps + (sstep + 1));
			_m_store(pw, _m_and(absmask_v32f, _m_sub(_a, _b))); pw += _m_inc;
			_m_store(pw, _m_and(absmask_v32f, _m_sub(_a, _c))); pw += _m_inc;
			_m_store(pw, _m_and(absmask_v32f, _m_sub(_a, _d))); pw += _m_inc;
			_m_store(pw, _m_and(absmask_v32f, _m_sub(_b, _c))); pw += _m_inc;
		}
		else for(k = sse; k--; ps += _m_inc) {
			_a = _m_loadu(ps), _b = _m_loadu(ps + 1), _c = _m_loadu(ps + sstep), _d = _m_loadu(ps + (sstep + 1));
			_m_store(pw, _m_and(absmask_v32f, _m_sub(_a, _b))); pw += _m_inc;
			_m_store(pw, _m_and(absmask_v32f, _m_sub(_a, _c))); pw += _m_inc;
			_m_store(pw, _m_and(absmask_v32f, _m_sub(_a, _d))); pw += _m_inc;
			_m_store(pw, _m_and(absmask_v32f, _m_sub(_b, _c))); pw += _m_inc;
		}
	}
	//process the last row
	if(srcAligned) for(k = sse; k--; ps += _m_inc, pw += _m_inc)
		_m_store(pw, _m_and(absmask_v32f, _m_sub(_m_load(ps), _m_loadu(ps + 1))));
	else for(k = sse; k--; ps += _m_inc, pw += _m_inc)
		_m_store(pw, _m_and(absmask_v32f, _m_sub(_m_loadu(ps), _m_loadu(ps + 1))));
	//the res pixels of last row
	for(k = 0; k < res; k++)
		*pw++ = yuabs(ps[k] - ps[k + 1]);
	//process the res pixels of each row (except the last row)
	ps = src + nsse;
	for(y = h1; y--; ps += sstep) {
		for(k = 0; k < res; k++) {
			*pw++ = yuabs(ps[k] - ps[k + 1]);
			*pw++ = yuabs(ps[k] - ps[k + sstep]);
			*pw++ = yuabs(ps[k] - ps[k + sstep + 1]);
			*pw++ = yuabs(ps[k + 1] - ps[k + sstep]);
		}
		*pw++ = yuabs(ps[k] - ps[k + sstep]);
	}

	//get corresponding edge nodes
	if(edgeNode[0] == numNodes) {
		int *pn = edgeNode;
#ifdef USE_AVX
#define LOOPinc(cmd) { cmd; cmd; cmd; cmd; cmd; cmd; cmd; cmd; }
#else
#define LOOPinc(cmd) { cmd; cmd; cmd; cmd; }
#endif
		ind = 0; //node index of current pixel
		for(y = h1; y--; ind += w - nsse) for(k = sse; k--;) {
			x = ind; LOOPinc(*pn++ = x++; *pn++ = x);
			x = ind; LOOPinc(*pn++ = x; *pn++ = x++ + w);
			x = ind; LOOPinc(*pn++ = x++; *pn++ = x + w);
			LOOPinc(*pn++ = ind + 1; *pn++ = ind++ + w);
		}
		for(k = sse; k--;)
			LOOPinc(*pn++ = ind++; *pn++ = ind);
		for(k = res; k--;) { 
			*pn++ = ind++; *pn++ = ind;
		}
		for(y = h1, ind = nsse; y--; ind += nsse) {
			for(k = res; k--;) {
				*pn++ = ind; *pn++ = ind + 1;
				*pn++ = ind; *pn++ = ind + w;
				*pn++ = ind; *pn++ = ind + w + 1;
				*pn++ = ind + 1; *pn++ = ind++ + w;
			}
			*pn++ = ind; *pn++ = ind++ + w;
		}
#undef LOOPinc
	}
#endif

	if(0) {
		//Check the correctness of edges
		for(k = 0; k < numEdges; k++) {
			_align_buf[0] = edgeWeight[k];
			x = edgeNode[k << 1]; y = edgeNode[(k << 1) + 1];
			_align_buf[1] = yuabs(src[x] - src[y]);
			if(_align_buf[0] != _align_buf[1] && 
				yuabs(_align_buf[0] - _align_buf[1]) > FLT_EPSILON)
				throw;
		}
	}

	//Sort edge weights in ascending order, and store the indexes
	//这个猜测是edgeWeight里面的顺序没变，但是index里面的索引都是升序以后的
	//比如说index里面的index[0]装的是edgeWeigt最小值对应的索引
	yuSortIdx(edgeWeight, index, numEdges);
	float maxWeight = edgeWeight[index[numEdges - 1]];
	if(maxWeight < 1e-4f) {
		//all the weights are zero, then all nodes goes to one component
		memset(root, 0, numNodes * sizeof(int));
		memset(labels, 0, numNodes * sizeof(int));
		areas[0] = numNodes;
		return 1;
	}

	//ind = int(numEdges * 9999.0 / 10000.0);
	//float segC = edgeWeight[index[ind]];
	//ctable0[k] = 1/k => 这个就是连通域vertex数目的一个倒数呗
	//ctable[k] = ctable0 * segC 
	float segC = .05f;
	yuMultiply(ctable0, segC, ctable, numNodes);

	//========== Core segmentation algorithm. ==========//

	//开始的时候每个node都相当于是一个root，应该是有多少个像素就要

	//Initially, each node (pixel) forms a segment (thus is a root).
	//If root[k] == k, then we think it is a root.
	//一开始每个像素位置都看作是一个连通域
	//如果root[k] == k, 那么就认为这个是一个根节点
	for(k = 0; k < numNodes; k++)
		root[k] = k;

	//结构体里面写的似乎是
	//thresh = max(edge_weigth) + c/size
	//但是这里赋值的segC明显就是上面的
	//float segC
	ROOTINFO roif; roif.size = 1; roif.thresh = segC;
	//这个只是分配一块连续的内存空间
	//应该是每个的size都为1，然后阈值都统一设置为了segC
	yuSetTo((uchar*)rootInfo, (uchar*)&roif, sizeof(ROOTINFO), numNodes);

	//Loop each edge.
	int num = numNodes, *nod;
	ROOTINFO *rf1, *rf2;
	float wt;
	for(k = 0; k < numEdges; k++) {
		//current edge index
		//indx里面是用来存放边界权重的
		//先取出edge权重最小的
		ind = index[k];
		//nodes corresponding to current edge
		//移位操作就是相当于把ind*2
		//这个其实就是把nod这个指针指向对应的位置
		nod = edgeNode + (ind << 1);
		//x和y就是我取出的，当前这个edge对应的两个vertex
		x = nod[0], y = nod[1];
		//find the root of node[x] & node[y]
		//如果我找到node[x]和node[y]的root，又符合和并条件，是不是就变成一个root了
		//这个是不是有那么一点点像heap了,这个root的逻辑和heap真的是挺像的
		//这个while结束的时候，x应该肯定就是root了
		while(root[x] != x)
			x = root[x];
		root[nod[0]] = x;
		while(root[y] != y)
			y = root[y];
		root[nod[1]] = y;
		if(x == y)
			continue;
		//join root[x] & root[y] by conditions
		//差不多是说node[0]和node[1]可能来自两个连通域
		//如果满足一定条件，就把他们所在的两个连通域给合并了
		wt = edgeWeight[ind];
		//这个是什么情况，x和y就是int类型的变量，可以直接
		//和结构体相加吗
		//不是这样的，这个其实是一个指针变量加上int，就相当于是
		//把这个指针移动了若干个单位
		rf1 = rootInfo + x, rf2 = rootInfo + y;
		//rootinfo里面的thresh应该是已经维护好了的，
		//就是总是确保max(edge_weight)
		if(wt < rf1->thresh && wt < rf2->thresh) {
			//这个是说总是大连通域合并小连通域吗
			if(rf1->size < rf2->size) {
				//把x的root给变成y，因为他们合并了
				root[x] = y;
				rf2->size += rf1->size;
				//ctable里面对应的每个元素都是乘上了segC的
				//thresh = max(edge_weight) + c/size
				//不是要求max(edge_weight)的吗，这里这个
				//是怎么确定的,比较以后应该能保证每次的wt都是最大的
				rf2->thresh = wt + ctable[rf2->size];
			}
			else {
				root[y] = x;
				rf1->size += rf2->size;
				rf1->thresh = wt + ctable[rf1->size];
			}
			num--; //number of segments (roots)
		}
	}

	//========== Post process. ==========//
	if(minArea == maxArea && maxArea == -101)
		return num;
	//Label each segment with continuous ID and get the size of each segment.
	if(minArea == maxArea) { //no area thresh
		x = 0; //id of each segment (root)
		for(k = 0; k < numNodes; k++)
			if(root[k] == k) {
				labels[k] = x;
				areas[x] = rootInfo[k].size;
				x++;
			}
		if(x != num) {
			printf("ImgSeg::segment() error: must be error in the code!");
			throw;
		}
	}
	else { //segments whose area are too small or too large will be set zero
		areas[0] = 0; x = 1;
		for(k = 0; k < numNodes; k++)
			if(root[k] == k) {
				y = rootInfo[k].size; //area
				if(y < minArea || y > maxArea) {
					labels[k] = 0;
					areas[0] += y;
				}
				else {
					labels[k] = x;//x是连续的，就是给连续的label
					areas[x] = y;
					x++;
				}
			}
		num = x;
	}
	if(num < numNodes)
		areas[num] = 0;
	//Generate the output labels image
	for(k = 0; k < numNodes; k++) if(root[k] != k) {
		//find the root of k-th node (pixel)
		ind = root[k];
		while(root[ind] != ind)
			ind = root[ind];
		//node k shares its root label
		labels[k] = labels[ind];
	}
	return num;
}

const unsigned char * getColor(int idx);

void ImgSeg::drawLabels(const int *label, uchar *img, int h, int w, int imStep)
{
	if(imStep)
		imStep -= w + w + w; //gap
	int x;
	for(; h--; img += imStep) for(x = w; x--; img += 3)
		memcpy(img, getColor(*label++), 3);
}

//=======================================================================================

int ImgSeg::segment2(const float *src, int sstep)
{
	if(!buffer.data) {
		printf("ImgSeg::segment2() error: please call init() first!");
		throw;
	}
	if(sstep == 0)
		sstep = w * 2; //sstep is default to: w * d
	if(sstep < w * 2) {
		printf("ImgSeg::segment2() error: sstep too small");
		throw;
	}

	//Get edge weights ------------------------------->>
	int k, y, x, ind;
	float *pw = edgeWeight;
	int sse = _m_nsse(w1), res = _m_res(w1), nsse = sse * _m_inc;
	const float *ps = src;
	const bool srcAligned = _m_aligned(size_t(ps) | size_t(sstep * sizeof(float)));
#ifdef USE_AVX
#define CMD \
	pw[0] = _align_buf[0] + _align_buf[1]; \
	pw[1] = _align_buf[2] + _align_buf[3]; \
	pw[2] = _align_buf[4] + _align_buf[5]; \
	pw[3] = _align_buf[6] + _align_buf[7]; \
	pw[4] = _align_buf[8] + _align_buf[9]; \
	pw[5] = _align_buf[10] + _align_buf[11]; \
	pw[6] = _align_buf[12] + _align_buf[13]; \
	pw[7] = _align_buf[14] + _align_buf[15]; \
	pw += 8;
#define LOOPinc(cmd) { cmd; cmd; cmd; cmd; cmd; cmd; cmd; cmd; }
#else
#define CMD \
	pw[0] = _align_buf[0] + _align_buf[1]; \
	pw[1] = _align_buf[2] + _align_buf[3]; \
	pw[2] = _align_buf[4] + _align_buf[5]; \
	pw[3] = _align_buf[6] + _align_buf[7]; \
	pw += 4;
#define LOOPinc(cmd) { cmd; cmd; cmd; cmd; }
#endif

//#define NORM_L1 //L1 or L2

#ifdef NORM_L1
#define OP_SSE(a,b,c,d) _m_store(_align_buf, _m_and(absmask_v32f, _m_sub(a, b))); _m_store(_align_buf + _m_inc, _m_and(absmask_v32f, _m_sub(c, d))); CMD
#define OP_FLT yuabs
#else
	__m _t;
#define OP_SSE(a,b,c,d) _t = _m_sub(a, b); _m_store(_align_buf, _m_mul(_t, _t)); _t = _m_sub(c, d); _m_store(_align_buf + _m_inc, _m_mul(_t, _t)); CMD
#define OP_FLT yusquare
#endif
	__m _a, _b, _c, _d, _e, _f, _g, _h;
	int gap = sstep - (nsse << 1);
	for(y = h1; y--; ps += gap)
		for(k = sse; k--;) {
			_a = _m_loadu(ps), _b = _m_loadu(ps + 2);
			_c = _m_loadu(ps + sstep), _d = _m_loadu(ps + (sstep + 2));
			ps += _m_inc;
			_e = _m_loadu(ps), _f = _m_loadu(ps + 2);
			_g = _m_loadu(ps + sstep), _h = _m_loadu(ps + (sstep + 2));
			ps += _m_inc;
			OP_SSE(_a, _b, _e, _f);
			OP_SSE(_a, _c, _e, _g);
			OP_SSE(_a, _d, _e, _h);
			OP_SSE(_b, _c, _f, _g);
		}
	//process the last row
	for(k = sse; k--;) {
		_a = _m_loadu(ps), _b = _m_loadu(ps + 2);
		ps += _m_inc;
		_e = _m_loadu(ps), _f = _m_loadu(ps + 2);
		ps += _m_inc;
		OP_SSE(_a, _b, _e, _f);
	}
	//the res pixels of last row
	res <<= 1;
	for(k = 0; k < res; k += 2)
		*pw++ = OP_FLT(ps[k] - ps[k + 2]) + OP_FLT(ps[k + 1] - ps[k + 3]);
	//process the res pixels of each row (except the last row)
	ps = src + (nsse << 1);
	for(y = h1; y--; ps += sstep) {
		for(k = 0; k < res; k += 2) {
			*pw++ = OP_FLT(ps[k] - ps[k + 2]) + OP_FLT(ps[k + 1] - ps[k + 3]);
			*pw++ = OP_FLT(ps[k] - ps[k + sstep]) + OP_FLT(ps[k + 1] - ps[k + sstep + 1]);
			*pw++ = OP_FLT(ps[k] - ps[k + sstep + 2]) + OP_FLT(ps[k + 1] - ps[k + sstep + 3]);
			*pw++ = OP_FLT(ps[k + 2] - ps[k + sstep]) + OP_FLT(ps[k + 3] - ps[k + sstep + 1]);
		}
		*pw++ = OP_FLT(ps[k] - ps[k + sstep]) + OP_FLT(ps[k + 1] - ps[k + sstep + 1]);
	}
	//--------------------------------------------------<<
	
	//Get corresponding edge nodes
	if(edgeNode[0] == numNodes) {
		int *pn = edgeNode;
		ind = 0; //node index of current pixel
		gap = w - nsse;
		for(y = h1; y--; ind += gap) for(k = sse; k--;) {
			x = ind; LOOPinc(*pn++ = x++; *pn++ = x);
			x = ind; LOOPinc(*pn++ = x; *pn++ = x++ + w);
			x = ind; LOOPinc(*pn++ = x++; *pn++ = x + w);
			LOOPinc(*pn++ = ind + 1; *pn++ = ind++ + w);
		}
		for(k = sse; k--;)
			LOOPinc(*pn++ = ind++; *pn++ = ind);
		for(k = 0; k < res; k += 2) {
			*pn++ = ind++; *pn++ = ind;
		}
		for(y = h1, ind = nsse; y--; ind += nsse) {
			for(k = 0; k < res; k += 2) {
				*pn++ = ind; *pn++ = ind + 1;
				*pn++ = ind; *pn++ = ind + w;
				*pn++ = ind; *pn++ = ind + w + 1;
				*pn++ = ind + 1; *pn++ = ind++ + w;
			}
			*pn++ = ind; *pn++ = ind++ + w;
		}
	}

	if(0) {
		//Check the correctness of edges
		for(k = 0; k < numEdges; k++) {
			_align_buf[0] = edgeWeight[k];
			x = edgeNode[k << 1]; y = edgeNode[(k << 1) + 1]; //node index
			x <<= 1; y <<= 1; //index in src[]
			_align_buf[1] = OP_FLT(src[x] - src[y]) + OP_FLT(src[x + 1] - src[y + 1]);
			if(_align_buf[0] != _align_buf[1] &&
				yuabs(_align_buf[0] - _align_buf[1]) > FLT_EPSILON)
				throw;
		}
	}

#undef CMD
#undef LOOPinc
#undef OP_SSE
#undef OP_FLT

	//Sort edge weights in ascending order, and store the indexes
	yuSortIdx(edgeWeight, index, numEdges);
	float maxWeight = edgeWeight[index[numEdges - 1]];
	if(maxWeight < 1e-4f) {
		//all the weights are zero, then all nodes goes to one component
		memset(root, 0, numNodes * sizeof(int));
		memset(labels, 0, numNodes * sizeof(int));
		areas[0] = numNodes;
		return 1;
	}
	
	//ind = int(numEdges * 9999.0 / 10000.0);
	//float segC = edgeWeight[index[ind]];
	float segC = 10.0f;//1.5f
	yuMultiply(ctable0, segC, ctable, numNodes);

	//========== Core segmentation algorithm. ==========//

	//Initially, each node (pixel) forms a segment (thus is a root).
	//If root[k] == k, then we think it is a root.
	for(k = 0; k < numNodes; k++)
		root[k] = k;

	ROOTINFO roif; roif.size = 1; roif.thresh = segC;
	yuSetTo((uchar*)rootInfo, (uchar*)&roif, sizeof(ROOTINFO), numNodes);

	//Loop each edge.
	int num = numNodes, *nod;
	ROOTINFO *rf1, *rf2;
	float wt;
	for(k = 0; k < numEdges; k++) {
		//current edge index
		ind = index[k];
		//nodes corresponding to current edge
		nod = edgeNode + (ind << 1);
		x = nod[0], y = nod[1];
		//find the root of node[x] & node[y]
		while(root[x] != x)
			x = root[x];
		root[nod[0]] = x;
		while(root[y] != y)
			y = root[y];
		root[nod[1]] = y;
		if(x == y)
			continue;
		//join root[x] & root[y] by conditions
		wt = edgeWeight[ind];
		rf1 = rootInfo + x, rf2 = rootInfo + y;
		if(wt < rf1->thresh && wt < rf2->thresh) {
			if(rf1->size < rf2->size) {
				root[x] = y;
				rf2->size += rf1->size;
				rf2->thresh = wt + ctable[rf2->size];
			}
			else {
				root[y] = x;
				rf1->size += rf2->size;
				rf1->thresh = wt + ctable[rf1->size];
			}
			num--; //number of segments (roots)
		}
	}

	//========== Post process. ==========//
	if(minArea == maxArea && maxArea == -101)
		return num;
	//Label each segment with continuous ID and get the size of each segment.
	if(minArea == maxArea) { //no area thresh
		x = 0; //id of each segment (root)
		for(k = 0; k < numNodes; k++)
			if(root[k] == k) {
				labels[k] = x;
				areas[x] = rootInfo[k].size;
				x++;
			}
		if(x != num) {
			printf("ImgSeg::segment2() error: must be error in the code!");
			throw;
		}
	}
	else { //segments whose area are too small or too large will be set zero
		areas[0] = 0; x = 1;
		for(k = 0; k < numNodes; k++)
			if(root[k] == k) {
				y = rootInfo[k].size; //area
				if(y < minArea || y > maxArea) {
					labels[k] = 0;
					areas[0] += y;
				}
				else {
					labels[k] = x;
					areas[x] = y;
					x++;
				}
			}
		num = x;
	}
	if(num < numNodes)
		areas[num] = 0;
	//Generate the output labels image
	for(k = 0; k < numNodes; k++) if(root[k] != k) {
		//find the root of k-th node (pixel)
		ind = root[k];
		while(root[ind] != ind)
			ind = root[ind];
		//node k shares its root label
		labels[k] = labels[ind];
	}
	return num;
}





/*void ImgSeg::segmentFlowImg(const float *Flow, const float *Img, int Fstep, int Istep)
{
	int _minArea = minArea, _maxArea = maxArea;
	minArea = maxArea = -101;

	extraBuf.reserve(numNodes * sizeof(int));
	int *root2 = (int*)extraBuf.p;

	segment2(Flow, Fstep);
	int k, x, y, z;
	for(k = 0; k < numNodes; k++) {
		x = k;
		while(root[x] != x)
			x = root[x];
		root[k] = x;
	}
	//yu::save("R2", yu::Mat(h, w, 1, yu::INT32, root));
	//save root[] table; root2 in [numNodes,numNodes2)
	//yuAdd(root, numNodes, root2, numNodes);
	//int numNodes2 = numNodes + numNodes;
	memcpy(root2, root, numNodes * sizeof(int));

	//root in [0,numNodes)
	segment(Img, Istep);
	for(k = 0; k < numNodes; k++) {
		x = k;
		while(root[x] != x)
			x = root[x];
		root[k] = x;
	}
	//yu::save("R", yu::Mat(h, w, 1, yu::INT32, root));
		
	//Merge the components denoted by root[] & root2[]
	for(k = 0; k < numNodes; k++) {
		//join segment(root[k]) & segment2(root2[k]) (cause they share the same node "k")
		x = y = k;
		//root of node k in segment()
		while(root[x] != x)
			x = root[x];
		//root[k] = x;
		//root of node k in segment2()
		while(root2[y] != y)
			y = root2[y];
		//root2[k] = y;
		//merge node[x] & node[y]
		//find the root of node[y]
		z = y;
		while(root[z] != z)
			z = root[z];
		//root[y] = z;
		if(x == z)
			continue;
		if(rootInfo[x].size < rootInfo[z].size) {
			root[x] = z;
			rootInfo[z].size += rootInfo[x].size;
		}
		else {
			root[z] = x;
			rootInfo[x].size += rootInfo[z].size;
		}
	}

	for(k = 0; k < numNodes; k++) {
		x = k;
		while(root[x] != x)
			x = root[x];
		root[k] = x;
	}
	//yu::save("Rf", yu::Mat(h, w, 1, yu::INT32, root));

	//========== Post process. ==========//
	//Label each segment with continuous ID and get the size of each segment.
	x = 0; //id of each segment (root)
	for(k = 0; k < numNodes; k++) if(root[k] == k) {
		labels[k] = x;
		areas[x] = rootInfo[k].size;
		x++;
	}
	//Generate the output labels image
	for(k = 0; k < numNodes; k++) if(root[k] != k) {
		//find the root of k-th node (pixel)
		int ind = root[k];
		while(root[ind] != k)
			ind = root[ind];
		//node k shares its root label
		labels[k] = labels[ind];
	}
}*/



int ImgSeg::mergeSegments(int **Roots, int n)
{
	int k = 0, x = 1, y, d, num = numNodes, *p;
	for(; k < numNodes; k++)
		root[k] = k;
	int *_areas = (int*)rootInfo; //rootInfo is not used in this function
	yuSetTo((uchar*)_areas, (uchar*)&x, sizeof(int), numNodes);

	//Set each node to its root, just to save computations
	for(d = 0; d < n; d++) {
		p = Roots[d];
		for(k = 0; k < numNodes; k++) if(p[k] != k) {
			x = p[k];
			while(p[x] != x)
				x = p[x];
			p[k] = x;
		}
	}

	//Loop each edge
	p = edgeNode;
	for(k = 0; k < numEdges; k++, p += 2) {
		//nodes corresponding to current edge: p[0] & p[1]
		//Loop each segmentation map
		for(d = 0; d < n; d++) {
			x = p[0], y = p[1];
			//find the root of node[x] & node[y] in the z-th segmentation map
			if(Roots[d][x] != x) //change "if" to "while"
				x = Roots[d][x];
			if(Roots[d][y] != y) //change "if" to "while"
				y = Roots[d][y];
			if(x != y)
				break;
		}
		if(d < n)
			continue;
		//Then merge node[p0] & node[p1]
		x = p[0], y = p[1];
		while(x != root[x])
			x = root[x];
		root[p[0]] = x;
		while(y != root[y])
			y = root[y];
		root[p[1]] = y;
		if(x == y)
			continue;
		if(_areas[x] < _areas[y]) {
			root[x] = y;
			_areas[y] += _areas[x];
		}
		else {
			root[y] = x;
			_areas[x] += _areas[y];
		}
		num--; //number of segments
	}

	//========== Post process. ==========//
	//Label each segment with continuous ID and get the size of each segment.
	if(minArea == maxArea) { //no area thresh
		x = 0; //start id of each segment (root)
		for(k = 0; k < numNodes; k++)
			if(root[k] == k) {
				labels[k] = x;
				areas[x] = _areas[k];
				x++;
			}
		if(x != num) {
			printf("ImgSeg::mergeSegments() error: must be error in the code!");
			throw;
		}
	}
	else { //segments whose area are too small or too large will be set zero
		areas[0] = 0; x = 1; //all background segments will be assigned 0
		for(k = 0; k < numNodes; k++)
			if(root[k] == k) {
				y = _areas[k];
				if(y < minArea || y > maxArea) {
					labels[k] = 0;
					areas[0] += y;
				}
				else {
					labels[k] = x;
					areas[x] = y;
					x++;
				}
			}
		num = x;
	}
	if(num < numNodes)
		areas[num] = 0;
	//Generate the output labels image
	for(k = 0; k < numNodes; k++) if(root[k] != k) {
		//find the root of k-th node (pixel)
		x = root[k];
		while(root[x] != x)
			x = root[x];
		//node k shares its root label
		labels[k] = labels[x];
	}
	return num;
}


int ImgSeg::segmentMulti(const float * const *srcs, int n)
{
	if(n == 1)
		return segment(srcs[0]);
	
	extraBuf.reserve(n * numNodes * sizeof(int) + n * sizeof(int*));
	int k, **Roots = (int**)(extraBuf.p + n * numNodes * sizeof(int));
	for(k = 0; k < n; k++)
		Roots[k] = k ? (Roots[k - 1] + numNodes) : (int*)extraBuf.p;
	//for(k = 0; k < n; k++)
	//	printf("%p\t%p\t%p\n", Roots + k, Roots[k], &Roots[k]);

	int _minArea = minArea, _maxArea = maxArea;
	minArea = maxArea = -101;

	for(k = 0; k < n; k++) {
		segment(srcs[k]);
		memcpy(Roots[k], root, numNodes * sizeof(int));
	}

	minArea = _minArea;
	maxArea = _maxArea;
	return mergeSegments(Roots, n);
}


int ImgSeg::segmentFlowImg(const float *Flow, const float *Img, int Fstep, int Istep)
{
	extraBuf.reserve(2 * numNodes * sizeof(int) + 2 * sizeof(int*));
	int **Roots = (int**)(extraBuf.p + 2 * numNodes * sizeof(int));
	Roots[0] = (int*)extraBuf.p;
	Roots[1] = Roots[0] + numNodes;

	int _minArea = minArea, _maxArea = maxArea;
	minArea = maxArea = -101;

	segment2(Flow, Fstep);
	memcpy(Roots[0], root, numNodes * sizeof(int));

	segment(Img, Istep);
	memcpy(Roots[1], root, numNodes * sizeof(int));

	minArea = _minArea;
	maxArea = _maxArea;
	return mergeSegments(Roots, 2);
}