#pragma once
#include <opencv2/core/mat.hpp>
#include "types.h"
#include <iostream>

struct HoughParams
{
	double pairwiseDist;
	double param1;
	double param2;
	int minRadius;
	int maxRadius;
};

class ShapeDetection
{
public:
	static CircleArray findHoughCircles(const cv::Mat& frame, const HoughParams& params);

	static RectangleArray findRectangles(const cv::Mat& frame);

};

void drawShape(cv::Mat& frame, const Circle& c);

void drawShape(cv::Mat& frame, const Rectangle& r);

template <typename SupportedShape>
void drawShapes(cv::Mat& frame, const std::vector<SupportedShape>& ShapeArray)
{
	for (const SupportedShape& shape : ShapeArray)
	{
		drawShape(frame, shape);
	}
}



