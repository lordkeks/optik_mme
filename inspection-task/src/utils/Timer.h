#pragma once
#include <chrono>
#include <string>
class Timer
{
public:
	using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

	static TimePoint tic();
	static void toc(const TimePoint& tp);
	static void toc(const TimePoint& tp, const std::string& funcname);
};

