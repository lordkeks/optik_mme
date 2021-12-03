#include "utils/helpers.h"
#include "utils/Timer.h"
#include "core/FrameGrabber.h"
#include "core/ShapeDetection.h"
#include "core/types.h"

#include <opencv2/opencv.hpp>


int main()
{
	auto src = cv::imread("./Download.jpg");

	cv::Mat frame;
	cv::cvtColor(src, frame, cv::COLOR_RGB2GRAY);

	HoughParams params;
	params.pairwiseDist = frame.size().height / 8;
	params.param1 = 110;
	params.param2 = 95;
	params.minRadius = 25;
	params.maxRadius = 110;

	auto circles = ShapeDetection::findHoughCircles(frame, params);
	auto tp = Timer::tic();
	drawShapes(src, circles);
	Timer::toc(tp);


	imshow("Window", src);
	cv::waitKey(0);

	cv::destroyAllWindows();
}