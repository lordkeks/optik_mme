#include "FrameGrabber.h"
#include <iostream>


FrameGrabber::FrameGrabber(int deviceidx)
	: vstream_(deviceidx)
{
	try 
	{
		if (!this->isValid())
		{
			throw std::invalid_argument("Couldn't connect to Device.\n");
		}
	}

	catch (const std::invalid_argument& e)
	{
		this->~FrameGrabber();
		std::cerr << "Exception: " << e.what();		
		exit(1);
	}
	
}

FrameGrabber::~FrameGrabber()
{
	this->vstream_.release();
}

bool FrameGrabber::isValid() const
{
	return this->vstream_.isOpened();
}

cv::Mat FrameGrabber::getFrame()
{
	cv::Mat frame;
	this->vstream_.read(frame);
	return frame;
}

MatArray FrameGrabber::getFrames(int number)
{
	MatArray frames;
	frames.reserve(number);
	for (int i = 0; i < number; i++)
	{
		auto frame = this->getFrame();
		frames.push_back(frame);
	}
	return frames;
}

cv::VideoCapture& FrameGrabber::getCaptureObject()
{
	return this->vstream_;
}
