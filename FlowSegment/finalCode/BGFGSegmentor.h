/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 12 of the book:
OpenCV3 Computer Vision Application Programming Cookbook
Third Edition
by Robert Laganiere, Packt Publishing, 2016.

This program is free software; permission is hereby granted to use, copy, modify,
and distribute this source code, or portions thereof, for any purpose, without fee,
subject to the restriction that the copyright notice may not be removed
or altered from any source or altered source distribution.
The software is released on an as-is basis and without any warranties of any kind.
In particular, the software is not guaranteed to be fault-tolerant or free from failure.
The author disclaims all warranties with regard to this software, any use,
and any consequent failure, is purely the responsibility of the user.

Copyright (C) 2016 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

#if !defined BGFGSeg
#define BGFGSeg

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/optflow.hpp>

#include <deque>

#include <chrono>
#include "mat_opencv.h"
#include "ImgSeg.h"
#include "videoprocessor.h"

using namespace std::chrono;

class BGFGSegmentor : public FrameProcessor {
	
	int minArea; 
	int maxArea;
public:

	BGFGSegmentor() : minArea{ 64 }, maxArea{ 50000 } {}

	void process(cv::Mat& current, cv::Mat& last, cv::Mat& output) {

		const int minMaxArea[2] = { minArea, maxArea };
		cv::cvtColor(current, current, cv::COLOR_BGR2GRAY);
		cv::cvtColor(last, last, cv::COLOR_BGR2GRAY);

		cv::Ptr<cv::optflow::DualTVL1OpticalFlow> tvl1 = cv::optflow::createOptFlow_DualTVL1();


		auto optflowStart = high_resolution_clock::now();
		cv::Mat flow;
		tvl1->calc(last, current, flow);
		auto optflowEnd = high_resolution_clock::now();
		auto durationOptflow = duration_cast<microseconds>(optflowEnd - optflowStart);
		std::cout << "optflow Sec: " << durationOptflow.count() / 1000000.0 << std::endl;


		yu::Mat draw;

		ImgSeg seger;
		seger.init(flow.rows, flow.cols, minMaxArea);
		const int smoothKernel = 7;
		const double sigma = 0.2 * ((smoothKernel - 1) * 0.5 - 1) + 0.4;

		cv::GaussianBlur(flow, flow, cv::Size(smoothKernel, smoothKernel), sigma, sigma);
		yu::Mat myflow = yu::get(flow);

		auto segmentStart = high_resolution_clock::now();
		int nFlowComps = seger.segment2((float*)myflow.p);
		auto segmentEnd = high_resolution_clock::now();
		auto durationSegment = duration_cast<microseconds>(segmentEnd - segmentStart);
		std::cout << "segment Sec: " << durationSegment.count() / 1000000.0 << std::endl;

		draw.create(myflow.h, myflow.w, 3, yu::UINT8);
		seger.drawLabels(seger.labels, draw.p, myflow.h, myflow.w);

		cv::Mat flowSeg = yu::get(draw);


		cv::Mat flowBinary;
		cv::cvtColor(flowSeg, flowBinary, cv::COLOR_BGR2GRAY);
		cv::threshold(flowBinary, flowBinary, 0, 255, cv::THRESH_BINARY);
		//先腐蚀后膨胀，去除一些细小的连接
		cv::erode(flowBinary, flowBinary, cv::Mat());
		cv::dilate(flowBinary, flowBinary, cv::Mat());

		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierachy;

		cv::findContours(flowBinary, contours, hierachy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));

		cv::Mat result_origin(flowBinary.size(), CV_8U, cv::Scalar(255));
		cv::drawContours(result_origin, contours,
			-1,
			cv::Scalar(0),
			1);

		
		cv::cvtColor(current, output, cv::COLOR_GRAY2BGR);

		for (const auto& contour : contours) {
			cv::Rect boundingBox = cv::boundingRect(contour);
			float aspect = boundingBox.height / boundingBox.width;
			if (cv::contourArea(contour) < 50)
				continue;
			cv::rectangle(output, boundingBox, cv::Scalar(0, 255, 0));
		}
	}

};

#endif
