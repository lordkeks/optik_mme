#pragma once
#include <vector>

#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>

using MatArray = std::vector<cv::Mat>;
class FrameGrabber
{
public:
	FrameGrabber(int deviceidx);
	~FrameGrabber();

	bool isValid() const;

	cv::Mat getFrame();
	MatArray getFrames(int number);

	cv::VideoCapture& getCaptureObject();


private:
	cv::VideoCapture vstream_;
};


