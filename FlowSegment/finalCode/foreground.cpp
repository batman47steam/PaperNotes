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

#include "videoprocessor.h"
#include "BGFGSegmentor.h"

int main()
{


	// Create video procesor instance
	VideoProcessor processor;

	// Create background/foreground segmentor 
	BGFGSegmentor segmentor;
	

	// Open video file
	processor.setInput("playground2.mp4");

	// set frame processor
	processor.setFrameProcessor(&segmentor);

	// Declare a window to display the video
	processor.displayOutput("Extracted Foreground");

	// Play the video at the original frame rate
	// 这个其实就是每帧之间的时间间隔

	std::cout << 1000. / processor.getFrameRate() << std::endl;
	processor.setDelay(1000./processor.getFrameRate());

	// Start the process
	processor.run();

	cv::waitKey();
} 