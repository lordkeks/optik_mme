#pragma once
#include "types.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>

namespace imgops
{
	bool isGrayscale(const cv::Mat& src);

	cv::Mat binarize(const cv::Mat& src, double thresh, double maxval);

	cv::Mat grayscaling(const cv::Mat& src);

}