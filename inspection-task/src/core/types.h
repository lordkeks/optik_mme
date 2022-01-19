#pragma once
#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv2/core/matx.hpp>

using Circle = cv::Vec3f;
using CircleArray = std::vector<Circle>;

using Rectangle = cv::Rect;
using RectangleArray = std::vector<Rectangle>;

using CvContour = std::vector<cv::Point2i>;
using CvContours = std::vector<CvContour>;

enum class CvColor
{
	RED = 0,
	BLUE = 1,
	GREEN = 2,
	YELLOW = 3,
	ORANGE = 4,
	MAGENTA = 5
};


inline cv::Scalar cvColorDescriptor(CvColor cflag)
{
	switch (cflag)
	{
	case CvColor::RED:
		return cv::Scalar(0, 0, 255);
	case CvColor::BLUE:
		return cv::Scalar(255, 0, 0);
	case CvColor::GREEN:
		return cv::Scalar(0, 255, 0);
	case CvColor::YELLOW:
		return cv::Scalar(0, 255, 255);
	case CvColor::ORANGE:
		return cv::Scalar(0, 165, 255);
	case CvColor::MAGENTA:
		return cv::Scalar(116, 0, 226);
	default:
		return cv::Scalar(0, 0, 0);
	}
}

