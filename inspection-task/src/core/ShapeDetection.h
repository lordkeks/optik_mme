#pragma once
#include <opencv2/core/mat.hpp>
#include "types.h"
#include <iostream>
#include <opencv2/imgproc.hpp>

struct FindCirclesOpts
{
	double pairwiseDist;
	double param1;
	double param2;
	int minRadius;
	int maxRadius;
};

struct FindRectsOpts
{
	int contoursRetrievalMode = cv::RETR_EXTERNAL;
	int contoursApproxMethod = cv::CHAIN_APPROX_SIMPLE;
};

class ShapeDetection
{
public:
	static CircleArray findHoughCircles(const cv::Mat& grayscale, const FindCirclesOpts& params);

	static RectangleArray findRectangles(const cv::Mat& bw, int minArea, 
										 const FindRectsOpts& params = FindRectsOpts());

};

void drawShape(cv::Mat& frame, const Circle& c, CvColor	clr);

void drawShape(cv::Mat& frame, const Rectangle& r, CvColor	clr);

void drawShape(cv::Mat& frame, const CvContours& cntrs, CvColor	clr);

template <typename SupportedShape>
void drawShapes(cv::Mat& frame, const std::vector<SupportedShape>& ShapeArray, CvColor clr)
{
	for (const SupportedShape& shape : ShapeArray)
	{
		drawShape(frame, shape, clr);
	}
}



