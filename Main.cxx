//headers
#include <cstdio>
#include <algorithm>
#include <functional>
//-//numbers
#include <cmath>
#include <limits>
#include <numeric>
//-//memory
#include <memory>
//-//strings
#include <string>
#include <string_view>
#include <fmt/format.h>
//-//input-output
#include <iostream>
//-//filesystem
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
//-//containers
#include <array>
#include <vector>
#include <unordered_map>
//-//debug
#include <exception>
//typedef
using tCmdKey = std::string_view;
using tCmdFun = std::function<void(tCmdKey)>;
using tCmdTab = std::unordered_map<tCmdKey, tCmdFun>;
//consdef
//datadef
static const tCmdTab cCmdTab{
	{"tFdirPathMain",
	 [](tCmdKey vCmdKey)
	 {
		 auto vPath = boost::filesystem::current_path();
		 fmt::println("[{0}]=({1})", vCmdKey, vPath.c_str());
	 }},
};
//actions
void fMain(std::vector<std::string_view> vArgV)
{
}//fMain
int main(int vArgC, char *vArgV[])
{
	try
	{
		if(vArgC <= 1)
		{
			throw std::runtime_error("invalid number of arguments");
		}
		else if(auto vI = cCmdTab.find(vArgV[1]); vI != cCmdTab.end())
		{
			vI->second(vI->first);
		}
		else
		{
            fMain({&vArgV[0], &vArgV[vArgC]});
		}
	}
	catch(const std::exception &rError)
	{
		fmt::println("we have an exception here: {0}", rError.what());
		return 1;
	}//catch(std::exception&)
	return 0;
}//main
