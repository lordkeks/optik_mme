#include "utils/helpers.h"
#include "utils/Timer.h"
#include "core/FrameGrabber.h"
#include "core/ShapeDetection.h"
#include "core/types.h"
#include "core/imageops.h"
#include <numeric>
#include <functional>


#include <opencv2/opencv.hpp>
#include <format>

class Container
{
public:
	using Callback = std::function<void(int a, int b)>;
	Container() = default;

	void registerCallback(const Callback& cb)
	{
		this->callback_ = nullptr;
		this->callback_ = cb;
	}

	bool callbackRegistered()
	{
		return this->callback_ != nullptr;
	}

	void call(int a, int b)
	{
		if (callbackRegistered())
		{
			callback_(a, b);
			return;
		}
		print("No Callbck");
		
	}

public:
	Callback callback_;
};


void calli(int a, int b)
{
	print("haey");
}


int main()
{
	fprint("HEy was geht {} {}", 3, 5);
	exit(0);
	Container cont;
	cont.call(1, 2);
	cont.registerCallback(calli);
	cont.call(1, 2);
	exit(0);

	
	auto src = cv::imread("./Download2.jpg");
	auto src2 = src.clone();

	auto grayscale = imgops::grayscaling(src);
	auto bw = imgops::binarize(src, 127, 255);

	FindCirclesOpts params;
	params.pairwiseDist = 5;
	params.param1 = 100;
	params.param2 = 100;
	params.minRadius = 5;
	params.maxRadius = 200;

	CircleArray circles = ShapeDetection::findHoughCircles(grayscale, params);
	drawShapes(src, circles, CvColor::YELLOW);

	//RectangleArray rects = ShapeDetection::findRectangles(bw, 3000);
	//drawShapes(src2, rects, CvColor::RED);


	cv::imshow("Circles", src);
	//cv::imshow("bw", src2);
	cv::waitKey(0);

	//drawShape(src, contours, CvColor::YELLOW);
	//cv::imshow("Window", dst);
	//cv::waitKey(0);

	//exit(0);
	//FindCirclesOpts params;
	//params.pairwiseDist = frame.size().height / 8;
	//params.param1 = 110;
	//params.param2 = 95;
	//params.minRadius = 25;
	//params.maxRadius = 110;

	//auto circles = ShapeDetection::findHoughCircles(frame, params);
	//auto tp = Timer::tic();
	//drawShapes(src, circles, CvColor::YELLOW);
	//Timer::toc(tp);


	//imshow("Window", src);
	//cv::waitKey(0);

	cv::destroyAllWindows();





}