#pragma once
#include <iostream>
#include <format>


template <typename T, typename... Args>
constexpr void print(T&& first, Args&&... rest) noexcept
{
	if constexpr (sizeof...(Args) == 0)
	{
		std::cout << first << "\n";               // for only 1-arguments
	}
	else
	{
		std::cout << first << ", ";        // print the 1 argument
		print(std::forward<Args>(rest)...); // pass the rest further
	}
}



#if ((defined(_MSVC_LANG) && _MSVC_LANG > 201703L) || __cplusplus > 201703L)
template <typename... Args>
constexpr void fprint(std::string_view frmt, const Args&... args)
{
	std::cout << std::format(frmt, args...) << "\n";
}
#else
	#define fprint print
#endif
