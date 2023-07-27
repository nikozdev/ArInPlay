//headers
#include <cstdio>
#include <algorithm>
#include <functional>
//-//numbers
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <Eigen/Dense>
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
//datadef
static std::random_device								vRandDevice;
static std::mt19937_64									vRandEngine(vRandDevice());
static std::uniform_real_distribution<> vRandNorm(-1.0, +1.0);
static std::uniform_int_distribution<>	vRandBool(0, 1);
//typedef
using tCmdKey = std::string_view;
using tCmdFun = std::function<void()>;
using tCmdTab = std::unordered_map<tCmdKey, tCmdFun>;
//-//maths
using tNum	= double;					//the type of number to use
using tVec	= Eigen::VectorXd;//array of numbers
using tMat	= Eigen::MatrixXd;//array of vectors
using tNode = tNum;						//input and output values
using tEdge = tNum;						//input-output coefficient
using tBias = tNum;						//aka bias
/* type of layer of neural network */
typedef class tLayerOfNetwork
{
public://codetor

	virtual ~tLayerOfNetwork() = default;

public://actions

	virtual void fAhead(tVec &vIputVec) = 0;
	virtual void fAback(tVec &vOputVec) = 0;

public://operats

	virtual std::ostream &operator<<(std::ostream &vStream) const = 0;

} tLayerOfNetwork;
inline std::ostream &
operator<<(std::ostream &vStream, const tLayerOfNetwork &rLayer)
{
	return rLayer.operator<<(vStream);
}//operator<<
/* type of layer of network dense */
typedef class tLayerOfNetworkDense final: public tLayerOfNetwork
{
public://codetor

	tLayerOfNetworkDense(size_t vIputDim, size_t vOputDim)
		: vNodeVec(vIputDim), vEdgeMat(vOputDim, vIputDim), vBiasVec(vOputDim)
	{
		for(size_t vY = 0; vY < vOputDim; vY++)
		{
			vBiasVec[vY] = vRandNorm(vRandEngine);
			for(size_t vX = 0; vX < vIputDim; vX++)
			{
				vEdgeMat(vY, vX) = vRandNorm(vRandEngine);
			}
		}
	}

public://actions

	virtual void fAhead(tVec &vIputVec) override
	{
		vNodeVec = vIputVec;
		vIputVec = (vEdgeMat * vIputVec) + vBiasVec;
	}//fAhead

	virtual void fAback(tVec &vOputVec) override
	{
		vEdgeMat = vEdgeMat - ((vOputVec * vNodeVec.transpose()) * 0.1);
		vBiasVec = vBiasVec - ((vOputVec)*0.1);
		vOputVec = vEdgeMat.transpose() * vOputVec;
	}//fAback

public://operats

	virtual std::ostream &operator<<(std::ostream &vStream) const override
	{
		vStream << "[NodeVec]=(" << std::endl;
		vStream << vNodeVec << std::endl;
		vStream << ")=[NodeVec]" << std::endl;
		vStream << "[EdgeMat]=(" << std::endl;
		vStream << vEdgeMat << std::endl;
		vStream << ")=[EdgeMat]" << std::endl;
		vStream << "[BiasVec]=(" << std::endl;
		vStream << vBiasVec << std::endl;
		vStream << ")=[BiasVec]" << std::endl;
		return vStream;
	}//operator<<

private://datadef

	tVec vNodeVec;//neuron vector

	tMat vEdgeMat;//weight matrix
	tVec vBiasVec;//bias vector

} tLayerOfNetworkDense;
/* type of layer of network activation */
typedef class tLayerOfNetworkActiv: public tLayerOfNetwork
{
public://typedef

	using tActiv = std::function<void(tVec &)>;

public://codetor

	tLayerOfNetworkActiv(const tActiv &fActiv, const tActiv &fPrime)
		: fActiv{fActiv}, fPrime{fPrime}
	{
	}

public://actions

	virtual void fAhead(tVec &vIputVec) override
	{
		vNodeVec = vIputVec;
		fActiv(vIputVec);
	}//fAhead
	virtual void fAback(tVec &vOputVec) override
	{
		fPrime(vNodeVec);
		vOputVec = vOputVec.array() * vNodeVec.array();
	}//fAback

public://operats

	virtual std::ostream &operator<<(std::ostream &vStream) const override
	{
		vStream << "[NodeVec]=(" << std::endl;
		vStream << vNodeVec << std::endl;
		vStream << ")=[NodeVec]" << std::endl;
		return vStream;
	}//operator<<

private://datadef

	tVec vNodeVec;

	tActiv fActiv, fPrime;

} tLayerOfNetworkActiv;
typedef class tLayerOfNetworkActivTanh final: public tLayerOfNetworkActiv
{
public://codetor

	tLayerOfNetworkActivTanh(): tLayerOfNetworkActiv(fActivVec, fPrimeVec)
	{
	}

public://actions

	static tNum fActivNum(tNum vIputNum)
	{
		return std::tanh(vIputNum);
	}//fActivNum
	static void fActivVec(tVec &vIputVec)
	{
		for(auto &vIputNum: vIputVec)
		{
			vIputNum = fActivNum(vIputNum);
		}
	}//fActivVec
	static tNum fPrimeNum(tNum vOputNum)
	{
		return (1 - std::pow(std::tanh(vOputNum), 2.0));
	}//fPrimeNum
	static void fPrimeVec(tVec &vOputVec)
	{
		for(auto &vOputNum: vOputVec)
		{
			vOputNum = fPrimeNum(vOputNum);
		}
	}//fPrimeVec

} tLayerOfNetworkActivTanh;
/* type of graph of neural network */
typedef class tGraphOfNetwork final
{
public://typedef

	using tLayer = tLayerOfNetwork;
	using tRefer = std::shared_ptr<tLayer>;
	using tArray = std::vector<tRefer>;

private://codetor

	tGraphOfNetwork() = default;

public://codetor

	~tGraphOfNetwork() = default;

public://actions

	inline void fAhead(tVec &vIputVec)
	{
		for(size_t vIndex = 0; vIndex < vArray.size(); vIndex++)
		{
			vArray[vIndex]->fAhead(vIputVec);
		}
	}//fAhead
	inline void fAback(tVec &vOput)
	{
		for(size_t vIndex = vArray.size(); --vIndex > 0; vIndex)
		{
			vArray[vIndex]->fAback(vOput);
		}
	}//fAback
	inline void fLearn(tVec &vIputVec, const tVec &vTrueVec)
	{
		fAhead(vIputVec);
		tVec vCostVec = (2 * (vIputVec - vTrueVec)).colwise().mean();
		fAback(vCostVec);
	}//fLearn

public://operats

	inline std::ostream &operator<<(std::ostream &vStream) const
	{
		for(size_t vIndex = 0; vIndex < vArray.size(); vIndex++)
		{
			vStream << "[" << vIndex << "]=(" << std::endl;
			vStream << *vArray[vIndex];
			vStream << ")=[" << vIndex << "]" << std::endl;
		}
		return vStream;
	}//operator<<

private://datadef

	tArray vArray;

private://friends

	typedef class tMakerOfNetwork tMakerOfNetwork;
	friend tMakerOfNetwork;

} tGraphOfNetwork;
inline std::ostream &
operator<<(std::ostream &vStream, const tGraphOfNetwork &rGraph)
{
	return rGraph.operator<<(vStream);
}//operator<<
/* type of maker of network */
typedef class tMakerOfNetwork final
{
public://typedef

	using tGraph = tGraphOfNetwork;

public://actions

	tMakerOfNetwork(): pGraph{new tGraph()}
	{
	}

public://actions

	template<typename tLayerT, typename... tArgT>
	auto fMakeLayer(tArgT &&...rArgT)
	{
		auto vLayer = std::make_shared<tLayerT>(std::forward<tArgT>(rArgT)...);
		this->pGraph->vArray.push_back(vLayer);
		return *this;
	}
	auto fTakeGraph()
	{
		return this->pGraph;
	}

private://datadef

	std::shared_ptr<tGraph> pGraph;

} tMakerOfNetwork;
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
static const tCmdTab cCmdTab{
	{"tFileSystem",
	 []()
	 {
		 auto vPath = nFileSystem::current_path();
		 nTextFormat::
			 println(stdout, "[{0:s}]=({1:s})", fPairTextWithCode(dPathToInternal));
		 nTextFormat::println(
			 stdout,
			 "[{0:s}]=({1:d})",
			 dPathToInternal,
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
	 }},
	{"tTextFormat",
	 []()
	 {
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
	 }},
	{
		"tLoopInversion", []()
		{
			for(unsigned vI = 2; vI > 0; --vI)
			{
				nTextFormat::println(stdout, "[I]=({})", vI);
			}
		}, },
	{"tMakerOfNetwork",
	 []()
	 {
		 auto pGraphOfNetwork
			 = tMakerOfNetwork()
					 .fMakeLayer<tLayerOfNetworkDense>(2, 3)
					 .fMakeLayer<tLayerOfNetworkActivTanh>()
					 .fMakeLayer<tLayerOfNetworkDense>(3, 1)
					 .fMakeLayer<tLayerOfNetworkActivTanh>()
					 .fTakeGraph();
		 std::clog << "[pGraphOfNetwork]=(" << std::endl;
		 std::clog << *pGraphOfNetwork << ")" << std::endl;
	 }},
	{"tSolutionOfXor",
	 []()
	 {
		 auto pGraphOfNetwork
			 = tMakerOfNetwork()
					 .fMakeLayer<tLayerOfNetworkDense>(2, 3)
					 .fMakeLayer<tLayerOfNetworkActivTanh>()
					 .fMakeLayer<tLayerOfNetworkDense>(3, 1)
					 .fMakeLayer<tLayerOfNetworkActivTanh>()
					 .fTakeGraph();
		 size_t vCount = 1'000'000;
		 for(size_t vIndex = 0; vIndex < vCount; vIndex++)
		 {
			 auto vInputL = static_cast<bool>(vRandBool(vRandEngine));
			 auto vInputR = static_cast<bool>(vRandBool(vRandEngine));
			 auto vInputV = tVec(2);
			 vInputV[0]		= static_cast<tNum>(vInputL);
			 vInputV[1]		= static_cast<tNum>(vInputR);
#if 0
       pGraphOfNetwork->fAhead(vInputV);
#else
			 auto vAnswer = tVec(1);
			 vAnswer[0]		= static_cast<tNum>(vInputL ^ vInputR);
			 pGraphOfNetwork->fLearn(vInputV, vAnswer);
#endif
			 if(vIndex % (vCount / 10) == 0)
			 {
				 if(vInputV[0] > 0.5)
				 {
					 vInputV[0] = 1.0;
				 }
				 else
				 {
					 vInputV[0] = 0.0;
				 }
				 std::clog << "[" << vInputL << " ^ " << vInputR << "]";
				 std::clog << " = " << vInputV << " " << vAnswer << std::endl;
			 }
		 }
	 }},
	{"tMatrix",
	 []()
	 {
		 tMat vM0(2, 2);
		 vM0(0, 0) = 2.0;
		 vM0(1, 1) = 2.0;
		 tMat vM1(2, 2);
		 vM1(0, 0) = 4.0;
		 vM1(1, 0) = 4.0;
		 std::clog << vM0 << std::endl << "*" << std::endl << vM1 << std::endl;
		 std::clog << "=" << std::endl << vM0 * vM1 << std::endl << std::endl;
		 std::clog << vM1 << std::endl << "*" << std::endl << vM0 << std::endl;
		 std::clog << "=" << std::endl << vM1 * vM0 << std::endl << std::endl;
		 tVec vV0(2);
		 std::clog << vM0 << std::endl << "*" << std::endl << vV0 << std::endl;
		 std::clog << "=" << std::endl << vM0 * vV0 << std::endl << std::endl;
		 std::clog << vV0 << std::endl << "*" << std::endl << vM1 << std::endl;
		 std::clog << "=" << std::endl << vV0 * vM1 << std::endl << std::endl;
	 }},
};
//getters
sf::Color fGetColor(float vValue, bool vFill, bool vSign)
{
	typedef union tPixel
	{
		struct
		{
			sf::Uint8 vA, vB, vG, vR;
		};
		sf::Uint32 vF = 0x00'00'00'00;
	} tPixel;
	sf::Uint8 vColorBase = vSign ? ((vValue + 1.0) * 60.0) : (vValue * 120.0);
	tPixel		vPixel;
	vPixel.vR = vColorBase;
	vPixel.vG = vColorBase;
	vPixel.vB = vColorBase;
	if(!vFill)
	{
		vPixel.vR = 0xff - vPixel.vR;
		vPixel.vG = 0xff - vPixel.vG;
		vPixel.vB = 0xff - vPixel.vB;
	}
	vPixel.vA = 0xff;
	return sf::Color{vPixel.vF};
}//fGetColor
 //actions
void fDraw(sf::RenderWindow &rWindow, const tDrawList &rDrawList)
{
}//fDraw
int fMain()
{
	//filesystem
	nFileSystem::current_path(dPathToInternal);
	fThrowIfNot(
		nFileSystem::current_path() == dPathToInternal,
		std::runtime_error(nTextFormat::format(
			"failed to find the internal path: {0}", dPathToInternal
		))
	);
	fThrowIfNot(
		nFileSystem::exists(dPathToResource),
		std::runtime_error(nTextFormat::format(
			"failed to find the resource path: {0}", dPathToResource
		))
	);
	//window
	const sf::VideoMode				vVideoMode(1'024, 1'024, 32);//sx,sy,bpp
	const auto								cStyle = sf::Style::Default; //bar|resize|close
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
	auto			pFont = std::make_shared<sf::Font>();
	fThrowIfNot(
		pFont->loadFromFile(dPathToResource "/kongtext.ttf"),
		std::runtime_error("failed font loading")
	);
	//timing
	sf::Clock vClock;
	sf::Time	vTimePNow = vClock.getElapsedTime();
	sf::Time	vTimePWas = vTimePNow;
	float			vTimeFNow = vTimePWas.asSeconds();
	float			vTimeFWas = vTimeFNow;
	unsigned	vTimeIWas = static_cast<unsigned>(vTimeFWas);
	unsigned	vTimeINow = static_cast<unsigned>(vTimeFNow);
	//intel
	auto pGraphOfNetwork
		= tMakerOfNetwork()
				.fMakeLayer<tLayerOfNetworkDense>(2, 3)
				.fMakeLayer<tLayerOfNetworkActivTanh>()
				.fMakeLayer<tLayerOfNetworkDense>(3, 1)
				.fMakeLayer<tLayerOfNetworkActivTanh>()
				.fTakeGraph();
	//mainloop
	while(vWindow.isOpen())
	{
		//timing
		vTimePWas					 = vTimePNow;
		vTimePNow					 = vClock.getElapsedTime();
		unsigned vTimeIWas = static_cast<unsigned>(vTimeFWas);
		vTimeFWas					 = vTimeFNow;
		vTimeFNow					 = vTimePNow.asSeconds();
		unsigned vTimeINow = static_cast<unsigned>(vTimeFNow);
		//intel
		if(static_cast<unsigned>(vTimePNow.asMilliseconds()) % 500 == 0)
		{
			auto vInputL = static_cast<bool>(vRandBool(vRandEngine));
			auto vInputR = static_cast<bool>(vRandBool(vRandEngine));
			auto vInputV = tVec(2);
			vInputV[0]	 = static_cast<tNum>(vInputL);
			vInputV[1]	 = static_cast<tNum>(vInputR);
			auto vAnswer = tVec(1);
			vAnswer[0]	 = static_cast<tNum>(vInputL ^ vInputR);
			pGraphOfNetwork->fLearn(vInputV, vAnswer);
		}//intel
		sf::Event vWindowEvent;
		vWindow.pollEvent(vWindowEvent);
		switch(vWindowEvent.type)
		{
		case sf::Event::Closed:
		{
			vWindow.close();
			break;
		}
		default: break;
		}
		vWindow.clear();
		for(tDrawIter vDrawIter: vDrawList)
		{
			vWindow.draw(*vDrawIter);
		}
		vWindow.display();
	}//mainloop
	return EXIT_SUCCESS;
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
			nTextFormat::println(stdout, "[{0:s}]=(", vI->first);
			vI->second();
			nTextFormat::println(stdout, ")=[{0:s}]", vI->first);
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
