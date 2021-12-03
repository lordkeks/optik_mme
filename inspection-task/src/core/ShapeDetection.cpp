#include "ShapeDetection.h"
#include <opencv2/imgproc.hpp>

CircleArray ShapeDetection::findHoughCircles(const cv::Mat& frame, const HoughParams& params)
{
	CircleArray circles;
	cv::HoughCircles(frame,
		circles,
		cv::HOUGH_GRADIENT,
		1,
		params.pairwiseDist,
		params.param1,
		params.param2,
		params.minRadius,
		params.maxRadius);
	return circles;
}

RectangleArray ShapeDetection::findRectangles(const cv::Mat& frame)
{
	//"Not Implemented yet"

}

void drawShape(cv::Mat& frame, const Circle& c)
{
	cv::Point2i center(static_cast<int>(c[0]), static_cast<int>(c[1]));
	auto radius = static_cast<int>(c[2]);
	// circle center
	circle(frame, center, 2, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
	// circle outline
	circle(frame, center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
}

void drawShape(cv::Mat& frame, const Rectangle& r)
{
	//"Not Implemented yet"

}

