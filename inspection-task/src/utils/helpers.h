#pragma once
#include <iostream>

template<typename Arg>
void print(const Arg& out, bool newline = true)
{
	if (newline)
	{
		std::cout << out << "\n";
	}
	else
	{
		std::cout << out;
	}
}