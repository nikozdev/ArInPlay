//headers
#include <cstdio>
#include <algorithm>
#include <functional>
//-//numbers
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
//-//memory
#include <memory>
//-//strings
#include <string>
#include <string_view>
#include <fmt/format.h>
#include <fmt/ranges.h>
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
//imports
namespace nFileSystem = boost::filesystem;
namespace nTextFormat = fmt;
//defines
#define fPairTextWithCode(vCode) #vCode, vCode
//-//debug
#define fDoIf(vExpr, vBool, ...) \
	({                             \
		if((vExpr) == vBool)         \
		{                            \
			__VA_ARGS__;               \
		}                            \
	})
#define fDoIfYes(vExpr, ...)					 fDoIf(vExpr, 1, __VA_ARGS__)
#define fDoIfNot(vExpr, ...)					 fDoIf(vExpr, 0, __VA_ARGS__)
#define fThrowIf(vExpr, vBool, vError) fDoIf(vExpr, vBool, throw vError)
#define fThrowIfYes(vExpr, vError)		 fDoIfYes(vExpr, throw vError)
#define fThrowIfNot(vExpr, vError)		 fDoIfNot(vExpr, throw vError)
//typedef
using tCmdKey = std::string_view;
using tCmdFun = std::function<void(tCmdKey)>;
using tCmdTab = std::unordered_map<tCmdKey, tCmdFun>;
//-//logic
using tNeuronValue = float;
using tNeuronLayer = std::vector<tNeuronValue>;
using tNeuronGraph = std::vector<tNeuronLayer>;
using tWeightValue = float;
using tWeightArray = std::vector<tWeightValue>;//from inputs
using tWeightLayer = std::vector<tWeightArray>;//into outputs
using tWeightGraph = std::vector<tWeightLayer>;
//-//graphics
using tDrawIter = std::shared_ptr<sf::Drawable>;
using tDrawList = std::vector<tDrawIter>;
//-//-//values
using tShapeValue = std::shared_ptr<sf::CircleShape>;
using tLabelValue = std::shared_ptr<sf::Text>;
using tJointValue = std::shared_ptr<sf::RectangleShape>;
//-//-//arrays
using tJointArray = std::vector<tJointValue>;
//-//-//layers
using tShapeLayer = std::vector<tShapeValue>;
using tLabelLayer = std::vector<tLabelValue>;
using tJointLayer = std::vector<tJointArray>;
//-//-//graphs
using tShapeGraph = std::vector<tShapeLayer>;
using tLabelGraph = std::vector<tLabelLayer>;
using tJointGraph = std::vector<tJointLayer>;
//consdef
//datadef
static const tCmdTab cCmdTab{
	{"tFileSystem",
	 [](tCmdKey vCmdKey)
	 {
		 auto vPath = nFileSystem::current_path();
		 nTextFormat::println(stdout, "[{0:s}]=(", vCmdKey);
		 nTextFormat::
			 println(stdout, "[{0:s}]=({1:s})", fPairTextWithCode(dPathToInternal));
		 nTextFormat::println(
			 stdout,
			 "[{0:s}]=({1:d})",
			 dPathToResource,
			 nFileSystem::exists(dPathToInternal)
		 );
		 nTextFormat::
			 println(stdout, "[{0:s}]=({1:s})", fPairTextWithCode(dPathToResource));
		 nTextFormat::println(
			 stdout,
			 "[{0:s}]=({1:d})",
			 dPathToResource,
			 nFileSystem::exists(dPathToResource)
		 );
		 nTextFormat::println(stdout, ")=[{0:s}]", vCmdKey);
	 }},
	{"tTextFormat",
	 [](tCmdKey vCmdKey)
	 {
		 nTextFormat::println(stdout, "[{0:s}]=(", vCmdKey);
		 nTextFormat::println(
			 stdout,
			 "[{0:s}]=({1:s})",
			 fPairTextWithCode(nTextFormat::format("{:.2f}", M_PI))
		 );
		 nTextFormat::println(
			 stdout,
			 "[{0:s}]=({1:s})",
			 fPairTextWithCode(nTextFormat::format("{:.02f}", M_PI))
		 );
		 nTextFormat::println(stdout, ")=[{0:s}]", vCmdKey);
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
	for(tDrawIter vDrawIter: rDrawList)
	{
		rWindow.draw(*vDrawIter);
	}
	rWindow.display();
}//fDraw
void fMain()
{
	std::random_device							 vRandomDevice;
	std::mt19937_64									 vRandomEngine(vRandomDevice());
	std::uniform_real_distribution<> vRandomNorm(-1.0, +1.0);
	std::uniform_int_distribution<>	 vRandomBool(0, 1);
	//filesystem
	nFileSystem::current_path(dPathToInternal);
	fThrowIfNot(
		nFileSystem::current_path() == dPathToInternal,
		std::runtime_error(nTextFormat::format(
			"failed to find the resource path: {0}", dPathToResource
		))
	);
	fThrowIfNot(
		nFileSystem::exists(dPathToResource),
		std::runtime_error(nTextFormat::format(
			"failed to find the resource path: {0}", dPathToResource
		))
	);
	//neural network
	tNeuronGraph vNGraph{
		{0, 1}, //input
		tNeuronLayer(4), //hidden
		{0.0}, //output
	};
	nTextFormat::println(stderr, "[NeuronGraph]=({0})", vNGraph);
	tWeightGraph vWGraph;
	size_t			 vWLayerCount = vNGraph.empty() ? 0 : ((vNGraph.size()) - 1);
	for(size_t vLIndex = 0; vLIndex < vWLayerCount; vLIndex++)
	{
		vWGraph.push_back({});
		auto &rWLayer	 = vWGraph.back();
		auto &rNLayerO = vNGraph[vLIndex + 1];
		auto &rNLayerI = vNGraph[vLIndex];
		for(size_t vAIndex = 0; vAIndex < rNLayerI.size(); vAIndex++)
		{
			rWLayer.push_back({});
			tWeightArray &rWArray = rWLayer.back();
			for(size_t vWIndex = 0; vWIndex < rNLayerO.size(); vWIndex++)
			{
				rWArray.push_back(vRandomNorm(vRandomEngine));
			}//create weight from each input into each output
			continue;
		}//create array of weights from each input into each output
		continue;
	}//create weight layer between each neuron layer
	nTextFormat::println(stderr, "[WeightGraph]=({0})", vWGraph);
	//window
	const sf::VideoMode				vVideoMode(1'024, 1'024, 32);//sx,sy,bpp
	const auto								cStyle = sf::Style::Default;
	const sf::ContextSettings vGfxSetup;
	sf::RenderWindow					vWindow(vVideoMode, "ArInPlay", cStyle, vGfxSetup);
	sf::Vector2f							vWindowSizeFull = {
		 static_cast<float>(vWindow.getSize().x),
		 static_cast<float>(vWindow.getSize().y),
	 };
	sf::Vector2f vWindowSizeHalf;
	vWindowSizeHalf.x = static_cast<float>(vWindowSizeFull.x) / 2.0;
	vWindowSizeHalf.y = static_cast<float>(vWindowSizeFull.y) / 2.0;
	//visual
	tDrawList vDrawList;
	//neurons
	tShapeGraph vSGraph;
	tLabelGraph vLGraph;
	auto				pFont = std::make_shared<sf::Font>();
	fThrowIfNot(
		pFont->loadFromFile(dPathToResource "/kongtext.ttf"),
		std::runtime_error("failed font loading")
	);
	auto vSStepX = vWindowSizeFull.x / (float)(vNGraph.size());
	for(size_t vLIndex = 0; vLIndex < vNGraph.size(); vLIndex++)
	{
		auto &rNLayer = vNGraph[vLIndex];
		vSGraph.push_back({});
		auto &rSLayer = vSGraph.back();
		vLGraph.push_back({});
		auto &rLLayer = vLGraph.back();
		auto	vSStepY = vWindowSizeFull.y / (float)(rNLayer.size());
		for(size_t vNIndex = 0; vNIndex < rNLayer.size(); vNIndex++)
		{
			tNeuronValue &rNValue		 = rNLayer[vNIndex];
			sf::Uint32		vColorBase = (rNValue + 1.0) * 60.0;
			sf::Uint32		vColorLine = 0xff'ff'ff'ff - vColorBase;
			sf::Uint32		vColorFill = 0xff'ff'ff'00 + vColorBase;
			//shape
			auto vRadius = vWindowSizeFull.x;
			vRadius /= (2 * vNGraph.size() * rNLayer.size());
			auto pSValue = std::make_shared<sf::CircleShape>(vRadius);
			pSValue->setOrigin(vRadius, vRadius);
			auto vCoord = vWindowSizeHalf;
			vCoord.x
				+= vSStepX * ((float)vLIndex + 0.375 - ((float)(vNGraph.size()) / 2.0));
			vCoord.y
				+= vSStepY * ((float)vNIndex + 0.5 - ((float)(rNLayer.size()) / 2.0));
			pSValue->setPosition(vCoord);
			rSLayer.push_back(pSValue);
			pSValue->setOutlineColor(sf::Color{vColorLine});
			pSValue->setFillColor(sf::Color{vColorFill});
			vDrawList.push_back(pSValue);
			//label
			auto pLValue = std::make_shared<sf::Text>();
			pLValue->setString(nTextFormat::format("{0:.2f}", rNValue));
			pLValue->setFont(*pFont);
			pLValue->setCharacterSize(vRadius / 4.0);
			pLValue->setPosition(vCoord);
			auto vLRect			= pLValue->getGlobalBounds();
			auto vLSizeFull = vLRect.getSize();
			auto vLSizeHalf = vLSizeFull;
			vLSizeHalf.x /= 2.0;
			vLSizeHalf.y /= 2.0;
			pLValue->setOrigin(vLSizeHalf);
			pLValue->setOutlineColor(sf::Color{vColorFill});
			pLValue->setFillColor(sf::Color{vColorLine});
			rLLayer.push_back(pLValue);
			//drawlist
			vDrawList.push_back(pLValue);
		}//create shapes and labels for each neuron
		continue;
	}//create shapes and labels for each layer
	tJointGraph vJGraph;
	for(size_t vLIndex = 0; vLIndex < vWGraph.size(); vLIndex++)
	{
		vJGraph.push_back({});
		tJointLayer	 &rJLayer	 = vJGraph.back();
		tShapeLayer	 &rSLayerI = vSGraph[vLIndex];
		tShapeLayer	 &rSLayerO = vSGraph[vLIndex + 1];
		tWeightLayer &rWLayer	 = vWGraph[vLIndex];
		for(size_t vAIndex = 0; vAIndex < rWLayer.size(); vAIndex++)
		{
			rJLayer.push_back({});
			tJointArray	 &rJArray	 = rJLayer.back();
			tShapeValue		pSValueI = rSLayerI[vAIndex];
			sf::Vector2f	vSPointI = pSValueI->getPosition();
			tWeightArray &rWArray	 = rWLayer[vAIndex];
			for(size_t vWIndex = 0; vWIndex < rWArray.size(); vWIndex++)
			{
				tWeightValue &rWValue		 = rWArray[vWIndex];
				sf::Uint32		vColorBase = (rWValue + 1.0) * 60.0;
				sf::Uint32		vColorLine = 0xff'ff'ff'ff - vColorBase;
				sf::Uint32		vColorFill = 0xff'ff'ff'00 + vColorBase;
				//joint
				rJArray.push_back(std::make_shared<sf::RectangleShape>());
				tJointValue	 pJValue		 = rJArray.back();
				tShapeValue	 pSValueO		 = rSLayerO[vWIndex];
				sf::Vector2f vSPointO		 = pSValueO->getPosition();
				float				 vOpposite	 = vSPointO.y - vSPointI.y;
				float				 vAdjacent	 = vSPointO.x - vSPointI.x;
				float				 vHypotenuse = 0.0;
				vHypotenuse += (vOpposite * vOpposite);
				vHypotenuse += (vAdjacent * vAdjacent);
				vHypotenuse = std::sqrt(vHypotenuse);
				float vSin	= vOpposite / vHypotenuse;
				float vArc	= std::asinf(vSin);
				float vDeg	= vArc * 180.0 / M_PI;
				pJValue->setSize({vHypotenuse, 8.0});
				pJValue->setOrigin({0.0, 8.0});
				pJValue->setPosition(pSValueI->getPosition());
				pJValue->setRotation(vDeg);
				pJValue->setOutlineColor(sf::Color{vColorLine});
				pJValue->setFillColor(sf::Color{vColorFill});
				vDrawList.push_back(pJValue);
			}//create weight shape from each input into each output
			continue;
		}//create weight shape array from each input into each output
		continue;
	}//create weight shape layer between each neuron layer
	sf::Clock vClock;
	sf::Time	vTimePNow = vClock.getElapsedTime();
	sf::Time	vTimePWas = vTimePNow;
	float			vTimeFNow = vTimePWas.asSeconds();
	float			vTimeFWas = vTimeFNow;
	unsigned	vTimeIWas = static_cast<unsigned>(vTimeFWas);
	unsigned	vTimeINow = static_cast<unsigned>(vTimeFNow);
	while(vWindow.isOpen())
	{
		//timing
		vTimePWas					 = vTimePNow;
		vTimePNow					 = vClock.getElapsedTime();
		unsigned vTimeIWas = static_cast<unsigned>(vTimeFWas);
		vTimeFWas					 = vTimeFNow;
		vTimeFNow					 = vTimePNow.asSeconds();
		unsigned vTimeINow = static_cast<unsigned>(vTimeFNow);
		if(static_cast<sf::Uint32>(vTimePNow.asMilliseconds()) % 100)
		{
			//neural-network-training
			auto vInputL	= static_cast<bool>(vRandomBool(vRandomEngine));
			auto vInputR	= static_cast<bool>(vRandomBool(vRandomEngine));
			vNGraph[0][0] = vInputL;
			vNGraph[0][1] = vInputR;
			//forward propagation
			for(size_t vLIndex = 0; vLIndex < vWGraph.size(); vLIndex++)
			{
				auto &rJLayer	 = vJGraph[vLIndex];		//joint
				auto &rWLayer	 = vWGraph[vLIndex];		//weight
				auto &rNLayerI = vNGraph[vLIndex + 0];//neuron input
				auto &rSLayerI = vSGraph[vLIndex + 0];//shape input
				auto &rLLayerI = vLGraph[vLIndex + 0];//label input
				auto &rNLayerO = vNGraph[vLIndex + 1];//neuron output
				auto &rSLayerO = vSGraph[vLIndex + 1];//shape output
				auto &rLLayerO = vLGraph[vLIndex + 1];//label output
				std::fill(rNLayerO.begin(), rNLayerO.end(), tNeuronValue{0});
				for(size_t vAIndex = 0; vAIndex < rWLayer.size(); vAIndex++)
				{
					auto &rNValueI		= rNLayerI[vAIndex];//neuron input
					auto &rSValueI		= rSLayerI[vAIndex];//shape input
					auto	vSColorBase = static_cast<sf::Uint32>((rNValueI + 1.0) * 60.0);
					auto	vSColorFill = 0xff'ff'ff'00 + vSColorBase;
					auto	vSColorLine = 0xff'ff'ff'ff - vSColorBase;
					rSValueI->setOutlineColor(sf::Color{vSColorLine});
					rSValueI->setFillColor(sf::Color{vSColorFill});
					auto &rLValueI = rLLayerI[vAIndex];//label input
					rLValueI->setString(nTextFormat::format("{:.2f}", rNValueI));
					rLValueI->setOutlineColor(sf::Color{vSColorFill});
					rLValueI->setFillColor(sf::Color{vSColorLine});
					auto &rWArray = rWLayer[vAIndex];//weight
					auto &rJArray = rJLayer[vAIndex];//joint
					for(size_t vWIndex = 0; vWIndex < rWArray.size(); vWIndex++)
					{
						if(vTimeIWas != vTimeINow)
						{
							nTextFormat::println(
								stderr, "[LI]={}[AI]={}[WI]={}", vLIndex, vAIndex, vWIndex
							);
						}
						auto &rWValue			= rWArray[vWIndex];//weight
						auto &rJValue			= rJArray[vWIndex];//joint
						auto	vJColorBase = static_cast<sf::Uint32>((rWValue + 1.0) * 60.0);
						auto	vJColorLine = 0xff'ff'ff'ff - vJColorBase;
						auto	vJColorFill = 0xff'ff'ff'00 + vJColorBase;
						rJValue->setOutlineColor(sf::Color{vJColorLine});
						rJValue->setFillColor(sf::Color{vJColorFill});
						auto &rNValueO = rNLayerO[vWIndex];//neuron output
						rNValueO			 = rNValueO + (rNValueI * rWValue);
					}//activate neurons on the next layer using their weights
					for(size_t vWIndex = 0; vWIndex < rWArray.size(); vWIndex++)
					{
						auto &rNValueO	 = rNLayerO[vWIndex];//neuron output
						rNValueO				 = 1.0 / (1.0 + std::exp(-rNValueO));
						auto &rSValueO	 = rSLayerO[vWIndex];//shape output
						auto vSColorBase = static_cast<sf::Uint32>((rNValueO + 1.0) * 60.0);
						auto vSColorFill = 0xff'ff'ff'00 + vSColorBase;
						auto vSColorLine = 0xff'ff'ff'ff - vSColorBase;
						rSValueO->setOutlineColor(sf::Color{vSColorLine});
						rSValueO->setFillColor(sf::Color{vSColorFill});
						auto &rLValueO = rLLayerO[vWIndex];//neuron output
						rLValueO->setString(nTextFormat::format("{:.2f}", rNValueO));
						rLValueO->setOutlineColor(sf::Color{vSColorFill});
						rLValueO->setFillColor(sf::Color{vSColorLine});
					}//apply activation function
					continue;
				}
				continue;
			}//forward propagation
			auto vAnswer = static_cast<float>(vInputL ^ vInputR);
			//backward propagation
			for(size_t vLIndex = 0; vLIndex < vWLayerCount; vLIndex++)
			{
			}//backward propagation
		}
		fProc(vWindow);
		fDraw(vWindow, vDrawList);
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
		nTextFormat::
			println(stderr, "we have an exception here: {0}", rError.what());
		return EXIT_FAILURE;
	}//catch(std::exception&)
	catch(...)
	{
		nTextFormat::
			println(stderr, "i have no idea what came out of that black box");
		return EXIT_FAILURE;
	}//catch(...)
	return EXIT_SUCCESS;
}//main
