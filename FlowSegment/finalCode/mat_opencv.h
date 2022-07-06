#pragma once
#include "mat_head.h"

#ifdef USE_OPENCV

//#include "cv.h"
//#include "highgui.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace yu
{
	inline cv::Mat get(const Mat &m)
	{
		int depth;
		switch(m.type) {
		case UINT8:
			depth = CV_8U; break;
		case INT8:
			depth = CV_8S; break;
		case UINT16:
			depth = CV_16U; break;
		case INT16:
			depth = CV_16S; break;
		case INT32:
			depth = CV_32S; break;
		case FLOAT32:
			depth = CV_32F; break;
		case DOUBLE64:
			depth = CV_64F; break;
		default:
			yuErr("yuGet(): unsupported m.type!");
		}
		return cv::Mat(m.h, m.w, CV_MAKE_TYPE(depth, m.d), m.p, (size_t)m.step);
	}

	inline yu::Mat get(const cv::Mat &m)
	{
		MatType type;
		switch(m.depth()) {
		case CV_8U:
			type = UINT8; break;
		case CV_8S:
			type = INT8; break;
		case CV_16U:
			type = UINT16; break;
		case CV_16S:
			type = INT16; break;
		case CV_32S:
			type = INT32; break;
		case CV_32F:
			type = FLOAT32; break;
		case CV_64F:
			type = DOUBLE64; break;
		default:
			yuErr("yuGet(): unknown m.depth()!");
		}
		return yu::Mat(m.rows, m.cols, m.channels(), type, (void*)m.data, (int)m.step);
	}

	template<typename T>
	inline cv::Rect_<T> get(yu::Rect_<T> rect)
	{
		return cv::Rect_<T>(rect.x, rect.y, rect.w, rect.h);
	}

	template<typename T>
	inline yu::Rect_<T> get(cv::Rect_<T> rect)
	{
		return yu::Rect_<T>(rect.x, rect.y, rect.width, rect.height);
	}
}

#endif // USE_OPENCV
