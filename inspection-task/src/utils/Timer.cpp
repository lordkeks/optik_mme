#include "Timer.h"
#include <iostream>

Timer::TimePoint Timer::tic()
{
	return std::chrono::high_resolution_clock::now();
}

void Timer::toc(const TimePoint& tp)
{
	auto now = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> ms_double = now - tp;
	std::cout << "[Finished in <" << ms_double.count() << ">ms]\n";
}

void Timer::toc(const TimePoint& tp, const std::string& funcname)
{
	auto now = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> ms_double = now - tp;
	std::cout << "[" << funcname << " took <" << ms_double.count() << ">ms]\n";
}

