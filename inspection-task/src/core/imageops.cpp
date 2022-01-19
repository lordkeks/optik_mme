#include "imageops.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>




namespace imgops
{

	bool isGrayscale(const cv::Mat& src)
	{
		return src.channels() == 1 ? true : false;
	}

	cv::Mat binarize(const cv::Mat& src, double thresh, double maxval)
	{
		cv::Mat dst;
		if (!isGrayscale(src)) {
			cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);
		}
		cv::threshold(dst, dst, thresh, maxval, cv::THRESH_BINARY);
		return dst;
	}


	cv::Mat grayscaling(const cv::Mat& src)
	{
		if (isGrayscale(src)) {
			return src;
		}
		cv::Mat dst;
		cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);
		return dst;
	}
}