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

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/bgsegm.hpp>

//#define IMAGE_SEQUENCE

#include "./src/algorithms/algorithms.h"


//#define DEBUG

/* Background Subtraction Methods */
auto algorithmsName = BGS_Factory::Instance()->GetRegisteredAlgorithmsName();

int main()
{

	std::string algorithmName = "PixelBasedAdaptiveSegmenter";
	//std::string algorithmName = "ViBe";
	//std::string algorithmName = "MixtureOfGaussianV2";

#ifdef IMAGE_SEQUENCE
	std::string sequence("input/in%6d.jpg");
#else
	std::string sequence("playground2.mp4");
#endif // IMAGE_SEQUENCE

	// Open the video file
	cv::VideoCapture capture(sequence);
	// check if video successfully opened
	if (!capture.isOpened())
		return 0;

	std::cout << "Running " << algorithmName << std::endl;
	auto bgs = BGS_Factory::Instance()->Create(algorithmName);

	// current video frame
	cv::Mat frame; 
	// foreground binary image
	cv::Mat foreground;
	// background image
	cv::Mat background;

	cv::namedWindow("Extracted Foreground");

	// The Mixture of Gaussian object
	// used with all default parameters
	//cv::Ptr<cv::BackgroundSubtractorMOG2> ptrMOG = cv::createBackgroundSubtractorMOG2();//500, 6, 0.9, 15
	
	// Extract object coordinates from the foreground
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierachy;

	//targets of N frames
	//std::vector<std::vector<cv::Point2f>> nFrameTargets;

	bool stop(false);

	static int frameNum = 0;
	// for all frames in video
	while (!stop) {


		// read next frame if any
		if (!capture.read(frame))
			break;


		//cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
		//bgs->setShowOutput(false);
		bgs->process(frame, foreground, background);

		cv::imwrite("foreground" + std::to_string(frameNum) + ".bmp", foreground);

		int ksize = 11;
		double sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8;
		//cv::GaussianBlur(frame, frame, cv::Size(ksize, ksize), sigma, sigma);

		// update the background
		// and return the foreground
		//ptrMOG->apply(frame, foreground, 0.005);//0.08


		//cv::imwrite("foreground" + std::to_string(frameNum) + ".bmp", foreground);

		//cv::GaussianBlur(foreground, foreground, cv::Size(ksize, ksize), sigma, sigma);
		// Complement the image
		//cv::threshold(foreground, foreground, 70, 255, cv::THRESH_BINARY);//15


		cv::GaussianBlur(foreground, foreground, cv::Size(ksize, ksize), sigma, sigma);
		// Complement the image
		cv::threshold(foreground, foreground, 15, 255, cv::THRESH_BINARY);


		//需要直观的显示一下mask的效果
		//蓝色的部分是foreground和foreground mask的交集
		//被红色框框住的，可以说是最终的结果
		/*cv::Mat foregroundColor = foreground.clone();
		cv::cvtColor(foreground, foregroundColor, cv::COLOR_GRAY2BGR);
		
		for(int i=0; i<foreground.rows; i++)
			for (int j = 0; j < foreground.cols; j++) {
				if (foreground_mask.at<uchar>(i, j) == 255)
					foregroundColor.at<cv::Vec3b>(i, j)[2] = 0;
			}*/

#ifdef  DEBUG

		cv::imwrite("frame" + std::to_string(frameNum) + ".bmp", frame);
		cv::imwrite("foreground" + std::to_string(frameNum) + ".bmp", foreground);

#endif //  DEBUG

		
		

		/*auto itc = contours.begin();
		while (itc != contours.end()) {
			cv::Rect boundingBox = cv::boundingRect(*itc);
			float aspect = boundingBox.height / boundingBox.width;
			if (cv::contourArea(*itc) < 9)
				itc = contours.erase(itc);
			else
				++itc;
		}*/

		/*cv::Mat result_origin(foreground.size(), CV_8U, cv::Scalar(255));
		cv::drawContours(result_origin, contours,
			-1,
			cv::Scalar(0),
			cv::FILLED);*/
		//cv::Mat foreground_dilate = foreground.clone();
		cv::Mat result_vertical;
		cv::Mat result_horizontal;
		cv::Mat verticalElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 7));
		cv::dilate(foreground, result_vertical, verticalElement, cv::Point(-1, -1), 1);
		cv::Mat horizontalElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 1));
		cv::dilate(result_vertical, result_horizontal, horizontalElement, cv::Point(-1, -1), 1);


		cv::findContours(result_horizontal, contours, hierachy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));

		//auto itc = contours.begin();

		/*while (itc != contours.end()) {
			cv::Rect boundingBox = cv::boundingRect(*itc);
			float aspect = boundingBox.height / boundingBox.width;
			if (cv::contourArea(*itc) < 9 || cv::contourArea(*itc) > 400 || aspect > 4)
				itc = contours.erase(itc);
			else
				++itc;
		}*/

		for (const auto& contour : contours) {
			cv::Rect boundingBox = cv::boundingRect(contour);
			float aspect = boundingBox.height / boundingBox.width;
			if (cv::contourArea(contour) < 9 || aspect > 4)
				continue;
			//cv::rectangle(frame, boundingBox, cv::Scalar(0,0,255));
		}

		//cv::imwrite("frame" + std::to_string(frameNum) + ".bmp", frame);
		//cv::imwrite("foregroundColor" + std::to_string(frameNum) + ".bmp", foregroundColor);
		// 我是要在result_dilate的基础上再检测一次contours才对
		// 要不然他还是针对foreground的检测结果来的
		/*contours.clear();
		hierachy.clear();
		cv::findContours(~result_horizontal, contours, hierachy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
		cv::Mat ultimate(foreground.size(), CV_8U, cv::Scalar(255));
		cv::drawContours(ultimate, contours,
			-1,
			cv::Scalar(0),
			1);

		for (const auto& contour : contours) {
			cv::Rect boundingBox = cv::boundingRect(contour);
			cv::rectangle(frame, boundingBox, cv::Scalar(0,255,0));
		}*/

		// show foreground and background·
		cv::imshow("Extracted Foreground", frame);

		frameNum++;

		// introduce a delay
		// or press key to stop
		if (cv::waitKey(10) >= 0)
			stop = true;
	}

	cv::waitKey();

} 