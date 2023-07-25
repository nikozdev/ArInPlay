//headers
#include <fmt/format.h>
//-//standard
#include <cstdio>
#include <exception>
//-//-//numbers
#include <cmath>
#include <limits>
#include <numeric>
//-//-//memory
#include <memory>
//-//-//strings
#include <string>
#include <string_view>
//-//-//input-output
#include <fstream>
#include <iostream>
//actions
int main(int vArgC, char *vArgV[])
{
	try
	{
        throw std::runtime_error("HelloError");
	}
	catch(const std::exception &rError)
	{
		fmt::println("we have got an exception: {0}", rError.what());
	}//catch(std::exception&)
	return 0;
}//main
