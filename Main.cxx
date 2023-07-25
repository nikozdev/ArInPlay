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
//-//graphics
#include <SFML/Graphics.hpp>
//typedef
using tCmdKey = std::string_view;
using tCmdFun = std::function<void(tCmdKey)>;
using tCmdTab = std::unordered_map<tCmdKey, tCmdFun>;
//-//graphics
using tDrawList = std::initializer_list<sf::Drawable *>;
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
void fProc(sf::RenderWindow &rWindow)
{
	for(sf::Event vEvent; rWindow.pollEvent(vEvent);)
	{
		switch(vEvent.type)
		{
		case sf::Event::Closed:
		{
			rWindow.close();
		}
		case sf::Event::Resized:
		{
			continue;
		}
		case sf::Event::TextEntered:
		{
			continue;
		}
		case sf::Event::KeyPressed:
		{
			continue;
		}
		case sf::Event::KeyReleased:
		{
			continue;
		}
		default: continue;
		}
	}//events
}//fProc
void fDraw(sf::RenderWindow &rWindow, const tDrawList &rDrawList)
{
	rWindow.clear();
	for(auto *pDrawable: rDrawList)
	{
		rWindow.draw(*pDrawable);
	}
	rWindow.display();
}//fDraw
void fMain()
{
	sf::Clock		 vClock;
	sf::RenderWindow vWindow(sf::VideoMode(512, 512), "ArInPlay");
	sf::Vector2f vWindowSizeHalf;
	vWindowSizeHalf.x = static_cast<float>(vWindow.getSize().x)/2.0;
	vWindowSizeHalf.y = static_cast<float>(vWindow.getSize().y)/2.0;
	//image
	sf::Texture vTexture;
	if(vTexture.loadFromFile("rsc/.LogoR16x16y.png") == 0)
	{
		throw std::runtime_error("failed image loading");
	}
	sf::Sprite vLogo(vTexture);
	sf::FloatRect vLogoRect	 = vLogo.getGlobalBounds();
	sf::Vector2f  vLogoSizeFull = vLogoRect.getSize();
	sf::Vector2f  vLogoSizeHalf;
    vLogoSizeHalf.x = vLogoSizeFull.x / 2.0f;
    vLogoSizeHalf.y = vLogoSizeFull.y / 2.0f;
	vLogo.setOrigin(vLogoSizeHalf);
	vLogo.setPosition(vWindowSizeHalf);
	//text
	sf::Font vFont;
	if(vFont.loadFromFile("rsc/kongtext.ttf") == 0)
	{
		throw std::runtime_error("failed font loading");
	}
	sf::Text	  vText("Hello World", vFont, 0x20);
	sf::FloatRect vTextRect	 = vText.getGlobalBounds();
	sf::Vector2f  vTextSizeFull = vTextRect.getSize();
	sf::Vector2f  vTextSizeHalf;
    vTextSizeHalf.x = vTextSizeFull.x / 2.0f;
    vTextSizeHalf.y = vTextSizeFull.y / 2.0f;
	vText.setOrigin(vTextSizeHalf);
	vText.setPosition(vWindowSizeHalf);
	//loop
	while(vWindow.isOpen())
	{
		sf::Time vTimeP = vClock.getElapsedTime();
		float	 vTimeF = vTimeP.asSeconds();
		vLogo.setRotation(vTimeF * 10.0);
		fProc(vWindow);
		fDraw(vWindow, {&vLogo, &vText});
	}//loop
}//fMain
int main(int vArgC, char *vArgV[])
{
	try
	{
		if(vArgC <= 1)
		{
			fMain();
		}
		else if(auto vI = cCmdTab.find(vArgV[1]); vI != cCmdTab.end())
		{
			vI->second(vI->first);
		}
		else
		{
			throw std::invalid_argument("invalid command line arguments");
		}
	}
	catch(const std::exception &rError)
	{
		fmt::println("we have an exception here: {0}", rError.what());
		return EXIT_FAILURE;
	}//catch(std::exception&)
	return EXIT_SUCCESS;
}//main
