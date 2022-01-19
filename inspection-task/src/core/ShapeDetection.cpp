#include "ShapeDetection.h"
#include "imageops.h"


CircleArray ShapeDetection::findHoughCircles(const cv::Mat& grayscale, const FindCirclesOpts& params)
{
	if (!imgops::isGrayscale(grayscale)) {
		std::cerr << "Input has to be grayscale\n";
		return {};
	}

	CircleArray circles;
	cv::HoughCircles(grayscale,
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

RectangleArray ShapeDetection::findRectangles(const cv::Mat& bw, int minArea, 
										      const FindRectsOpts& params /*= FindRectsOpts()*/)
{
	CvContours contours;
	cv::findContours(bw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	RectangleArray fittingsrects;
	for (const CvContour& cntr : contours)
	{
		Rectangle rect = cv::boundingRect(cntr);
		if (rect.area() >= minArea) {
			fittingsrects.push_back(rect);
		}
	}
	return fittingsrects;
}

void drawShape(cv::Mat& frame, const Circle& c, CvColor	clr)
{
	auto color = cvColorDescriptor(clr);
	cv::Point2i center(static_cast<int>(c[0]), static_cast<int>(c[1]));
	auto radius = static_cast<int>(c[2]);
	// circle center
	circle(frame, center, 2, color, 3, cv::LINE_AA);
	// circle outline
	circle(frame, center, radius, color, 3, cv::LINE_AA);
}


void drawShape(cv::Mat& frame, const Rectangle& r, CvColor clr)
{
	cv::rectangle(frame, r, cvColorDescriptor(clr), 2);
}

void drawShape(cv::Mat& frame, const CvContours& cntrs, CvColor	clr)
{
	cv::drawContours(frame, cntrs, -1, cvColorDescriptor(clr), 2);
}

