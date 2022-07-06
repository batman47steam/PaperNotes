#pragma once

//Implement the "Efficient Graph-Based Image Segmentation" algorithm.
//by: yuxianguo_chn@163.com, 20180215.

#include "yustd.h"

class ImgSeg
{
public:
	int h, w, h1, w1, numNodes, numEdges; //numNodes = h*w
	int *root; //[numNodes], root[k] is the root node index of node[k]
	struct ROOTINFO {
		int size; //size of the segment
		float thresh; //thresh = max(edge_weight) + c/size
	};
	ROOTINFO *rootInfo; //[numNodes], each root corresponds to a segment, which has its size and thresh
	float *ctable0; //[numNodes+1], ctable0[k] = 1 / k,可能是为了快速的算C/size啥的
	float *ctable; //[numNodes+1], ctable0 * segC，segC应该就是控制阈值的那个
	float *edgeWeight; //[numEdges]
	int *edgeNode; //[numEdges*2], each edge corresponds to two nodes
	int *index; //[numEdges], edge weights are sorted in ascending order
	int minArea, maxArea; //segments whose area are out of the range will be labeled with 0

	Buffer buffer;
	void init(int rows, int cols, const int areaRange[2] = NULL);

	//Input single channel float type image (h-by-w);
	//Output single channel integer type segmentation image (h-by-w).
	//Note: The output labels are continuous, each segment has a unique label.
	//      The total number of output segments is returned, and the size of segment(i) is index[i].
	int segment(const float *src, int sstep = 0);
	int *labels; //[numNodes], return the segmentation result image
	int *areas; //[num], where the "num" is the number of segments in the segmentation image (num <= numNodes), return the size of each segment

public:
	//map continuously valued labels to a color image
	//label: [h*w], valued in 0:1:max
	//img: [h*w*3], output color image, has a withstep of imStep
	static void drawLabels(const int *label, uchar *img, int h, int w, int imStep = 0);

public:
	//Assume the input image has two channels, and of size "rows-by-cols".
	//The output are all the same as segment().
	int segment2(const float *src, int sstep = 0);


public:
	//Input an optical flow image (rows-cols-2) and an gray image (rows-cols-1).
	//Fstep & Istep are the row steps of Flow & Img respectively.
	//Only output the labels.
	//We will put two neighboring pixels in the same component when they appear in
	//the same component either Flow.segmentation or Img.segmentation.
	//void segmentFlowImg(const float *Flow, const float *Img, int Fstep = 0, int Istep = 0);
	Buffer extraBuf; //used only for segmentFlowImg()


public:
	//We will put two neighboring pixels in the same component only when
	//they appear in the same segment in all the given segmentation maps.

	//Merge a set of n segmentation maps.
	//Each Roots[i] should have numNodes vals, all be in [0,numNodes).
	//The output are all the same as segment().
	int mergeSegments(int **Roots, int n);

	//Segment a set of source images (all have the size h-w-1).
	int segmentMulti(const float * const *srcs, int n);

	//Segment a flow image and a gray image.
	int segmentFlowImg(const float *Flow, const float *Img, int Fstep = 0, int Istep = 0);
};