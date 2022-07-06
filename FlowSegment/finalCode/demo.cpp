#include "mat_opencv.h"
#include "ImgSeg.h"
#include <opencv2/optflow.hpp>

using namespace yu;

int main(){
    
    const int minMaxArea[2] = { 64, 50000 };

    cv::Mat frame_last;
    cv::Mat frame_current;

    cv::VideoCapture capture("playground2.mp4");

    if (!capture.isOpened())
        return 0;

    capture.read(frame_last);
    for (auto i = 0; i < 7; i++)
        capture.read(frame_current);
    cv::cvtColor(frame_last, frame_last, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame_current, frame_current, cv::COLOR_BGR2GRAY);


    // Create the optical flow algorithm
    cv::Ptr<cv::optflow::DualTVL1OpticalFlow> tvl1 = cv::optflow::createOptFlow_DualTVL1();

                                                                                                                // compute the optical flow between 2 frames
    cv::Mat flow; // image of 2D flow vectors
    // compute optical flow between frame1 and frame2
    tvl1->calc(frame_last, frame_current, flow);

    /*cv::Mat flow_two[2];
    cv::Mat flow;
    flow_two[0] = cv::imread("flow_1.bmp", 0);
    flow_two[1] = cv::imread("flow_2.bmp", 0);
    cv::merge(flow_two[2], flow);*/
    yu::Mat draw;
    
    //这个算法里面好像还没看到可调节的参数k
    ImgSeg seger;
    seger.init(flow.rows, flow.cols, minMaxArea);
    const int smoothKernel = 7;
    const double sigma = 0.2 * ((smoothKernel - 1) * 0.5 - 1) + 0.4;

    cv::GaussianBlur(flow, flow, cv::Size(smoothKernel, smoothKernel), sigma, sigma);
    //cv::Mat => yu::Mat
    yu::Mat myflow = yu::get(flow);

    int nFlowComps = seger.segment2((float*)myflow.p);
    draw.create(myflow.h, myflow.w, 3, yu::UINT8);
    seger.drawLabels(seger.labels, draw.p, myflow.h, myflow.w);

    //get根据传入mat类型的不同，可以实现cv::Mat=>y::Mat之间的双向转换
    cv::Mat flowSeg = yu::get(draw);
    //应该也又接口可以把yu::Mat,变成opencv的mat
    cv::imwrite("FlowSeg.bmp", yu::get(draw));

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

    cv::Mat frame_show;
    cv::cvtColor(frame_current, frame_show, cv::COLOR_GRAY2BGR);

    for (const auto& contour : contours) {
        cv::Rect boundingBox = cv::boundingRect(contour);
        float aspect = boundingBox.height / boundingBox.width;
        if (cv::contourArea(contour) < 50)
            continue;
        cv::rectangle(frame_show, boundingBox, cv::Scalar(0, 255, 0));
    }



    //拟合最小外接矩形
    std::vector<std::vector<cv::Point2i>> everyLabel(nFlowComps);
    int* labelPoint = seger.labels;
    //这里这样写是因为label也按照这个顺序来的吗
    for(int y=0; y < flow.rows; y++)
        for (int x = 0; x < flow.cols; x++) {
            everyLabel[*labelPoint++].push_back(cv::Point2i(x, y));
        }

    
    /*for (auto i = 1; i < nFlowComps; i++) {
        cv::Rect boundingBox = cv::boundingRect(everyLabel[i]);
        float aspect = boundingBox.height / boundingBox.width;
        if (aspect < 1 || aspect > 4)
            continue;
        cv::rectangle(frame_show, boundingBox, cv::Scalar(0, 255, 0));
    }*/
    return 0;

}