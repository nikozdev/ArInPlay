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
//-//execution
#include <future>
#include <thread>
//-//debug
#include <exception>
#include <chrono>
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
//getters
static auto fGetColor(float vValue, bool vFill)
{
	union tPixel
	{
		struct
		{
			sf::Uint8 vA, vB, vG, vR;
		};
		sf::Uint32 vF = 0x00'00'00'00;
	} vPixel;
	sf::Uint8 vColorBase = (vValue + 1.0) * 120.0;
	if(vFill)
	{
		vPixel.vR = vColorBase;
		vPixel.vG = vColorBase;
		vPixel.vB = vColorBase;
	}
	else
	{
		vPixel.vR = 0xff - vColorBase;
		vPixel.vG = 0xff - vColorBase;
		vPixel.vB = 0xff - vColorBase;
	}
	vPixel.vA = 0xff;
	return sf::Color{vPixel.vF};
}//fGetColor
//datadef
//-//random
static std::random_device								vRandDevice;
static std::mt19937_64									vRandEngine(vRandDevice());
static std::uniform_real_distribution<> vRandNorm(-1.0, +1.0);
static std::uniform_int_distribution<>	vRandBool(0, 1);
//typedef
template<typename tValue>
using tRefer = std::shared_ptr<tValue>;
//-//command
using tCmdKey = std::string_view;
using tCmdFun = std::function<void()>;
using tCmdTab = std::unordered_map<tCmdKey, tCmdFun>;
//-//graphics
using tJointValue = sf::RectangleShape;
using tJointArray = std::vector<tJointValue>;
using tJointTable = std::vector<tJointArray>;
using tShapeValue = sf::CircleShape;
using tShapeArray = std::vector<tShapeValue>;
//-//maths
using tNum	= float;					//the type of number to use
using tVec	= Eigen::VectorXd;//array of numbers
using tMat	= Eigen::MatrixXd;//array of vectors
using tNode = tNum;						//input and output values
using tEdge = tNum;						//input-output coefficient
using tBias = tNum;						//aka bias
//-//intel
class tMakerOfNetwork;
/* type of layer of neural network */
class tLayerOfNetwork
{
public://codetor

	virtual ~tLayerOfNetwork() = default;

public://actions

	virtual void fAhead(tVec &vIputVec) = 0;
	virtual void fAback(tVec &vOputVec) = 0;

	virtual void fDraw(sf::RenderWindow &vRender, const sf::Transform &vTform) = 0;

public://operats

	virtual std::ostream &operator<<(std::ostream &vStream) const = 0;

protected://friends

	friend class tMakerOfNetwork;

};//tLayerOfNetwork
inline std::ostream &
operator<<(std::ostream &vStream, const tLayerOfNetwork &rLayer)
{
	return rLayer.operator<<(vStream);
}//operator<<
/* type of layer of network dense */
class tLayerOfNetworkDense final: public tLayerOfNetwork
{
private://codetor

	tLayerOfNetworkDense(size_t vIputDim, size_t vOputDim, tNum vRateNum = 0.1)
		: vEdgeMat(vOputDim, vIputDim)
		, vNodeVec(vIputDim)
		, vBiasVec(vOputDim)
		, vRateNum(vRateNum)
		, vJointTable(vOputDim, tJointArray(vIputDim))
		, vShapeArray(vIputDim)
	{
		fThrowIfNot(
			vIputDim > 0,
			std::logic_error("layers must have at least input 1 dimension")
		);
		fThrowIfNot(
			vOputDim > 0,
			std::logic_error("layers must have at least output 1 dimension")
		);
		tNum vIScale = 1.00f / static_cast<tNum>(vIputDim);
		tNum vIStepY = vIScale;
		tNum vIFromX = 0.00f, vIFromY = 0.50f * vIScale;
		tNum vOScale = 1.00f / static_cast<tNum>(vOputDim);
		tNum vOStepY = vOScale;
		tNum vOFromY = 0.50f * vOScale, vOFromX = 1.0f / 6.0f;
		for(size_t vO = 0u; vO < vOputDim; vO++)
		{
			vBiasVec[vO]			= vRandNorm(vRandEngine);
			auto	vOposX			= vOFromX;
			auto	vOposY			= vOFromY + vOStepY * vO;
			auto &vJointArray = vJointTable[vO];
			for(size_t vI = 0u; vI < vIputDim; vI++)
			{
				vEdgeMat(vO, vI) = vRandNorm(vRandEngine);
				//coord
				auto vIposX = vIFromX;
				auto vIposY = vIFromY + vIStepY * vI;
				//angle
				auto vOpo = vOposY - vIposY;
				auto vAdj = vOposX - vIposX;
				auto vHyp = 0.0f;
				vHyp			= vHyp + (vOpo * vOpo);
				vHyp			= vHyp + (vAdj * vAdj);
				vHyp			= std::sqrt(vHyp);
				auto vSin = vOpo / vHyp;
				auto vArc = std::asinf(vSin);
				auto vDeg = vArc * 180.0f / M_PI;
				//sizes
				auto vShapeHalfs = std::min(vIScale * 0.1f, 0.02f);
				auto vJointSizes = sf::Vector2f{vHyp, 0.025f};
				vJointSizes.y		 = vJointSizes.y * std::min(vIScale, vOScale);
				vJointSizes.y		 = std::min(vJointSizes.y, 0.01f);
				auto vJointHalfs = vJointSizes;
				vJointHalfs.x		 = vJointHalfs.x * 0.5f;
				vJointHalfs.x		 = vJointHalfs.x * 0.5f;
				//joints
				auto &vJointValue = vJointArray[vI];
				vJointValue.setOrigin(0.0f, vJointHalfs.y);
				vJointValue.setPosition(vIposX, vIposY);
				vJointValue.setSize(vJointSizes);
				vJointValue.setRotation(vDeg);
				vJointValue.setOutlineThickness(vJointHalfs.y * 0.5f);
				vJointValue.setOutlineColor(fGetColor(1.0f, 1u));
				vJointValue.setFillColor(fGetColor(1.0f, 0u));
			}
		}//weights and biases
		for(size_t vI = 0; vI < vIputDim; vI++)
		{
			//coord
			auto vIposX = vIFromX;
			auto vIposY = vIFromY + vIStepY * vI;
			//sizes
			auto vShapeHalfs = std::min(vIScale * 0.1f, 0.01f);
			//shape
			auto &vShapeValue = vShapeArray[vI];
			vShapeValue.setOrigin(vShapeHalfs, vShapeHalfs);
			vShapeValue.setPosition(vIposX, vIposY);
			vShapeValue.setRadius(vShapeHalfs);
			vShapeValue.setOutlineThickness(vShapeHalfs * 0.1f);
			vShapeValue.setOutlineColor(fGetColor(1.0f, 0u));
			vShapeValue.setFillColor(fGetColor(1.0f, 1u));
		}//nodes
		return;
	}//tLayerOfNetworkDense

public://actions

	virtual void fAhead(tVec &vIputVec) override
	{
		vNodeVec = vIputVec;
		vIputVec = vEdgeMat * vNodeVec + vBiasVec;
	}//fAhead
	virtual void fAback(tVec &vOputVec) override
	{//transposition makes vector/matrix shapes matching
		vEdgeMat = vEdgeMat - vOputVec * vRateNum * vNodeVec.transpose();
		vBiasVec = vBiasVec - vOputVec * vRateNum;
		vOputVec = vEdgeMat.transpose() * vOputVec;
	}//fAback

	virtual void
	fDraw(sf::RenderWindow &vRender, const sf::Transform &vTform) override
	{
		for(size_t vO = 0; vO < vBiasVec.size(); vO++)
		{
			auto &vJointArray = vJointTable[vO];
			for(size_t vI = 0; vI < vNodeVec.size(); vI++)
			{
				auto &vJointValue = vJointArray[vI];
				auto	vJointColor = vEdgeMat(vO, vI);
				vJointValue.setOutlineColor(fGetColor(vJointColor, 1));
				vJointValue.setFillColor(fGetColor(vJointColor, 0));
				vRender.draw(vJointValue, vTform);
				continue;
			}//edge inputs
			continue;
		}//edge outputs
		for(size_t vI = 0; vI < vNodeVec.size(); vI++)
		{
			auto &vShapeValue = vShapeArray[vI];
			auto	vShapeColor = vNodeVec[vI];
			vShapeValue.setOutlineColor(fGetColor(vShapeColor, 0));
			vShapeValue.setFillColor(fGetColor(vShapeColor, 1));
			vRender.draw(vShapeValue, vTform);
		}//neurons
		return;
	}//fDraw

public://operats

	virtual std::ostream &operator<<(std::ostream &vStream) const override
	{
		vStream << "[EdgeMat]=(" << std::endl;
		vStream << vEdgeMat << std::endl;
		vStream << ")=[EdgeMat]" << std::endl;
		vStream << "[NodeVec]=(" << std::endl;
		vStream << vNodeVec << std::endl;
		vStream << ")=[NodeVec]" << std::endl;
		vStream << "[BiasVec]=(" << std::endl;
		vStream << vBiasVec << std::endl;
		vStream << ")=[BiasVec]" << std::endl;
		vStream << "[RateNum]=(" << std::endl;
		vStream << vRateNum << std::endl;
		vStream << ")=[RateNum]" << std::endl;
		return vStream;
	}//operator<<

private://datadef

	tMat vEdgeMat;//weight matrix
	tVec vNodeVec;//neuron vector
	tVec vBiasVec;//bias vector

	tNum vRateNum;//learning rate

	tJointTable vJointTable;
	tShapeArray vShapeArray;

private://friends

	friend class tMakerOfNetwork;

};//tLayerOfNetworkDense
/* type of layer of network activation */
template<auto fActiv, auto fPrime>
class tLayerOfNetworkActiv final: public tLayerOfNetwork
{
public://codetor

	tLayerOfNetworkActiv(size_t vIputDim)
		: vNodeVec(vIputDim)
		, vIShapeArray(vIputDim)
		, vOShapeArray(vIputDim)
		, vJointArray(vIputDim)
	{
		fThrowIfNot(
			vIputDim > 0,
			std::logic_error("layers must have at least input 1 dimension")
		);
		tNum vIScale = 1.00f / static_cast<tNum>(vIputDim);
		tNum vIFromX = 0.00f, vIFromY = 0.50f * vIScale;
		tNum vIStepY = vIScale;
		tNum vOScale = vIScale;
		tNum vOFromY = vIFromY, vOFromX = 1.0f / 6.0f;
		tNum vOStepY = vOScale;
		for(size_t vI = 0u; vI < vIputDim; vI++)
		{
			auto vOposX = vOFromX;
			auto vOposY = vOFromY + vOStepY * vI;
			auto vIposX = vIFromX;
			auto vIposY = vIFromY + vOStepY * vI;
			//shape
			auto	vIShapeHalfs = std::min(vIScale * 0.05f, 0.01f);
			auto &vIShapeValue = vIShapeArray[vI];
			vIShapeValue.setOrigin(vIShapeHalfs, vIShapeHalfs);
			vIShapeValue.setPosition(vIposX, vIposY);
			vIShapeValue.setRadius(vIShapeHalfs);
			vIShapeValue.setOutlineThickness(vIShapeHalfs * 0.010f);
			vIShapeValue.setOutlineColor(fGetColor(1.0f, 1u));
			vIShapeValue.setFillColor(fGetColor(1.0f, 0u));
			auto	vOShapeHalfs = std::min(vOScale * 0.05f, 0.01f);
			auto &vOShapeValue = vOShapeArray[vI];
			vOShapeValue.setOrigin(vOShapeHalfs, vOShapeHalfs);
			vOShapeValue.setPosition(vOposX, vOposY);
			vOShapeValue.setRadius(vOShapeHalfs);
			vOShapeValue.setOutlineThickness(vOShapeHalfs * 0.1f);
			vOShapeValue.setOutlineColor(fGetColor(1.0f, 1u));
			vOShapeValue.setFillColor(fGetColor(1.0f, 0u));
			//joint
			auto vJointSizes	= sf::Vector2f{vOposX - vIposX, 0.025f};
			vJointSizes.y			= vJointSizes.y * std::min(vIScale, vOScale);
			vJointSizes.y			= std::min(vJointSizes.y, 0.01f);
			auto vJointHalfs	= vJointSizes;
			vJointHalfs.x			= vJointHalfs.x * 0.5f;
			vJointHalfs.y			= vJointHalfs.y * 0.5f;
			auto &vJointValue = vJointArray[vI];
			vJointValue.setOrigin(0.0f, vJointHalfs.y);
			vJointValue.setPosition(vIposX, vIposY);
			vJointValue.setSize(vJointSizes);
			vJointValue.setOutlineThickness(vJointHalfs.y * 0.5f);
			vJointValue.setOutlineColor(fGetColor(1.0f, 0u));
			vJointValue.setFillColor(fGetColor(1.0f, 1u));
		}//nodes
		return;
	}//tLayerOfNetworkActiv

public://actions

	virtual void fAhead(tVec &vIputVec) override
	{
		vNodeVec = vIputVec;
		for(auto &vIputNum: vIputVec)
		{
			vIputNum = fActiv(vIputNum);
		}
	}//fAhead
	virtual void fAback(tVec &vOputVec) override
	{
		for(auto &vNodeNum: vNodeVec)
		{
			vNodeNum = fPrime(vNodeNum);
		}
		vOputVec = vOputVec.array() * vNodeVec.array();
	}//fAback

	virtual void
	fDraw(sf::RenderWindow &vRender, const sf::Transform &vTform) override
	{
		for(size_t vI = 0; vI < vNodeVec.size(); vI++)
		{
			//joint
			auto &vJointValue = vJointArray[vI];
			auto	vJointColor = vNodeVec[vI];
			vJointValue.setOutlineColor(fGetColor(vJointColor, 0));
			vJointValue.setFillColor(fGetColor(vJointColor, 1));
			vRender.draw(vJointValue, vTform);
			//shape
#if 1
			auto &vIShapeValue = vIShapeArray[vI];
			auto	vIShapeColor = vNodeVec[vI];
			vIShapeValue.setOutlineColor(fGetColor(vIShapeColor, 0));
			vIShapeValue.setFillColor(fGetColor(vIShapeColor, 1));
			vRender.draw(vIShapeValue, vTform);
#endif
#if 1
			auto &vOShapeValue = vOShapeArray[vI];
			auto	vOShapeColor = vNodeVec[vI];
			vOShapeValue.setOutlineColor(fGetColor(vOShapeColor, 0));
			vOShapeValue.setFillColor(fGetColor(vOShapeColor, 1));
			vRender.draw(vOShapeValue, vTform);
#endif
		}
	}//fDraw

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

	tJointArray vJointArray;
	tShapeArray vIShapeArray;
	tShapeArray vOShapeArray;

private://friends

	friend class tMakerOfNetwork;

};//tLayerOfNetworkActiv
	/* type of layer of network activation linear
	 * essentially it does not change anything
	 * this is like not having activation function at all
	 * in comparison with tanh it gives me terrible results
	 * > even 1k times more iterations of learning did not make it right
	 */
using tLayerOfNetworkActivLine = tLayerOfNetworkActiv<
	[](tNum vNum)
	{
		return vNum;
	},
	[](tNum vNum)
	{
		return 1.0;
	}>;//tLayerOfNetworkActivLine
/* type of layer of network activation sigmoid
 * this one actually sucks in comparison with hyperbolic tangent
 * > it needs 100 times more iterations to start solving xor
 */
using tLayerOfNetworkActivSigma = tLayerOfNetworkActiv<
	[](tNum vNum)
	{
		return 1.0 / (1.0 + std::exp(-vNum));
		//return 1.0 * std::tanh(vNum / 2.0) / 2.0;
	},
	[](tNum vNum)
	{
		//return (1.0 - std::pow(std::tanh(0.5 * vNum), 2.0)) / 2.0;
		tNum vSig = 1.0 / (1.0 + std::exp(-vNum));
		return vSig * (1.0 - vSig);
		//return (std::exp(-vNum)) / (1.0 + std::exp(-vNum));
	}>;//tLayerOfNetworkActivSigma
/* type of layer of network activation tangent hyperbolic */
using tLayerOfNetworkActivTanh = tLayerOfNetworkActiv<
	[](tNum vNum)
	{
		return std::tanh(vNum);
	},
	[](tNum vNum)
	{
		return (1.0 - std::pow(std::tanh(vNum), 2.0));
	}>;//tLayerOfNetworkActivTanh
/* type of layer of network activation rectangular linear unit */
using tLayerOfNetworkActivRelu = tLayerOfNetworkActiv<
	[](tNum vNum)
	{
		return vNum > 0.0 ? vNum : vNum * 0.1;
	},
	[](tNum vNum)
	{
		return vNum > 0.0 ? 1.0 : 0.1;
	}>;//tLayerOfNetworkActivRelu
/* type of graph of neural network */
class tGraphOfNetwork final
{
public://typedef

	using tLayerValue = tLayerOfNetwork;
	using tLayerRefer = tRefer<tLayerValue>;
	using tLayerArray = std::vector<tLayerRefer>;

private://codetor

	tGraphOfNetwork() = default;

private://actions

	auto fPushLayer(tLayerRefer vLayerRefer)
	{
		vLayerArray.push_back(vLayerRefer);
		return *this;
	}

public://actions

	inline void fAhead(tVec &vIputVec)
	{
		for(size_t vIndex = 0; vIndex < vLayerArray.size();)
		{
			vLayerArray[vIndex]->fAhead(vIputVec);
			vIndex++;
		}
	}//fAhead
	inline void fAback(tVec &vOput)
	{
		for(size_t vIndex = vLayerArray.size(); vIndex > 0;)
		{
			vIndex--;
			vLayerArray[vIndex]->fAback(vOput);
		}
	}//fAback
	inline void fLearn(tVec &vIputVec, const tVec &vTrueVec)
	{
		fAhead(vIputVec);
		tVec vCostVec = 2.0 * (vIputVec - vTrueVec);//calculate suqared error prime
		vCostVec			= vCostVec / static_cast<tNum>(vIputVec.size());//make it mean
		fAback(vCostVec);
	}//fLearn

	inline void fDraw(sf::RenderWindow &vWindow, sf::Transform &vTform)
	{
		tNum vStepX = 1.0f / static_cast<tNum>(vLayerArray.size());
		for(size_t vIndex = 0u; vIndex < vLayerArray.size(); vIndex++)
		{
			vLayerArray[vIndex]->fDraw(vWindow, vTform);
			vTform.translate({vStepX, 0.0f});
		}
	}//fDraw

public://operats

	inline std::ostream &operator<<(std::ostream &vStream) const
	{
		for(size_t vIndex = 0; vIndex < vLayerArray.size(); vIndex++)
		{
			vStream << "[" << vIndex << "]=(" << std::endl;
			vStream << *vLayerArray[vIndex];
			vStream << ")=[" << vIndex << "]" << std::endl;
		}
		return vStream;
	}//operator<<

private://datadef

	tLayerArray vLayerArray;

private://friends

	friend class tMakerOfNetwork;

};//tGraphOfNetwork
inline std::ostream &
operator<<(std::ostream &vStream, const tGraphOfNetwork &rGraph)
{
	return rGraph.operator<<(vStream);
}//operator<<
/* type of maker of network
 * class responsible for the network construction and handling
 */
class tMakerOfNetwork final
{
public://typedef

	using tGraphValue = tGraphOfNetwork;
	using tGraphRefer = std::shared_ptr<tGraphValue>;

public://codetor

	tMakerOfNetwork(): vGraphRefer{new tGraphValue()}
	{
	}

public://actions

	template<typename tLayerT, typename... tArgT>
	auto fMakeLayer(tArgT &&...rArgT)
	{
		auto vLayer = tRefer<tLayerT>(new tLayerT(std::forward<tArgT>(rArgT)...));
		vGraphRefer->fPushLayer(vLayer);
		return *this;
	}//fMakeLayer
	auto fTakeGraph()
	{
		return this->vGraphRefer;
	}//fTakeGraph

private://datadef

	tGraphRefer vGraphRefer;

};//tMakerOfNetwork
//actions
template<typename tVal>
tVal fRead(nFileSystem::ifstream &vFile)
{
	tVal					 vVal;
	constexpr auto cSize = sizeof(tVal);
	vFile.read(reinterpret_cast<char *>(&vVal), cSize);
#ifdef __APPLE__
	static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");
	union
	{
		tVal		vFull;
		uint8_t vByte[cSize];
	} vSource, vTarget;
	//convert
	vSource.vFull = vVal;
	for(size_t vI = 0; vI < cSize; vI++)
	{
		vTarget.vByte[vI] = vSource.vByte[cSize - vI - 1];
	}
	return vTarget.vFull;
#else
	return vVal;
#endif
}//fRead
//testing
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
	{"tFileReaderPushback",
	 []()
	 {
		 auto vTimeSince = std::chrono::high_resolution_clock::now();
		 auto vDataPath	 = dPathToResource "/mnist-train-images.idx3-ubyte";
		 auto vDataFile	 = nFileSystem::ifstream(vDataPath, std::ios::binary);
		 fThrowIfNot(
			 vDataFile.is_open(),
			 std::logic_error(
				 nTextFormat::format("failed to load the file: {}", vDataPath)
			 )
		 );
		 auto vDataPack = std::vector<uint8_t>();
		 for(uint8_t vDataItem; !vDataFile.eof(); vDataFile >> vDataItem)
		 {
			 vDataPack.push_back(vDataItem);
		 }
		 auto vTimeUntil = std::chrono::high_resolution_clock::now();
		 auto vTimeTaken = vTimeUntil - vTimeSince;
		 nTextFormat::print(
			 stdout,
			 "[TimeTaken]=(\n"
			 "[nanos]=({:L})"
			 "[micro]=({:L})"
			 "[milli]=({:L})"
			 "[secon]=({:L})"
			 ")=[TimeTaken]\n"
			 "[Data]=(\n"
			 "[Size]=({:L})"
			 ")=[Data]\n",
			 duration_cast<std::chrono::nanoseconds>(vTimeTaken).count(),
			 duration_cast<std::chrono::microseconds>(vTimeTaken).count(),
			 duration_cast<std::chrono::milliseconds>(vTimeTaken).count(),
			 duration_cast<std::chrono::seconds>(vTimeTaken).count(),
			 vDataPack.size()
		 );
	 }},
	{"tFileReaderIterator",
	 []()
	 {
		 auto vTimeSince = std::chrono::high_resolution_clock::now();
		 auto vDataPath	 = dPathToResource "/mnist-train-images.idx3-ubyte";
		 auto vDataFile	 = nFileSystem::ifstream(vDataPath, std::ios::binary);
		 fThrowIfNot(
			 vDataFile.is_open(),
			 std::logic_error(
				 nTextFormat::format("failed to load the file: {}", vDataPath)
			 )
		 );
		 auto vDataPack = std::vector<uint8_t>(
			 std::istreambuf_iterator<decltype(vDataFile)::char_type>(vDataFile),
			 std::istreambuf_iterator<decltype(vDataFile)::char_type>()
		 );
		 auto vTimeUntil = std::chrono::high_resolution_clock::now();
		 auto vTimeTaken = vTimeUntil - vTimeSince;
		 nTextFormat::print(
			 stdout,
			 "[TimeTaken]=(\n"
			 "[nanos]=({:L})"
			 "[micro]=({:L})"
			 "[milli]=({:L})"
			 "[secon]=({:L})"
			 ")=[TimeTaken]\n"
			 "[Data]=(\n"
			 "[Size]=({:L})"
			 ")=[Data]\n",
			 duration_cast<std::chrono::nanoseconds>(vTimeTaken).count(),
			 duration_cast<std::chrono::microseconds>(vTimeTaken).count(),
			 duration_cast<std::chrono::milliseconds>(vTimeTaken).count(),
			 duration_cast<std::chrono::seconds>(vTimeTaken).count(),
			 vDataPack.size()
		 );
	 }},
	{"tFileReaderPreallocSingle",
	 []()
	 {
		 auto vTimeSince = std::chrono::high_resolution_clock::now();
		 auto vDataPath	 = dPathToResource "/mnist-train-images.idx3-ubyte";
		 auto vDataFile	 = nFileSystem::ifstream(vDataPath, std::ios::binary);
		 fThrowIfNot(
			 vDataFile.is_open(),
			 std::logic_error(
				 nTextFormat::format("failed to load the file: {}", vDataPath)
			 )
		 );
		 size_t vDataSize = vDataFile.seekg(0, std::ios::end).tellg();
		 auto		vDataPack = std::vector<uint8_t>(vDataSize);
		 vDataFile.seekg(0, std::ios::beg);
		 for(size_t vI = 0; vI < vDataSize; vI++)
		 {
			 vDataFile >> vDataPack[vI];
		 }
		 auto vTimeUntil = std::chrono::high_resolution_clock::now();
		 auto vTimeTaken = vTimeUntil - vTimeSince;
		 nTextFormat::print(
			 stdout,
			 "[TimeTaken]=(\n"
			 "[nanos]=({:L})"
			 "[micro]=({:L})"
			 "[milli]=({:L})"
			 "[secon]=({:L})"
			 ")=[TimeTaken]\n"
			 "[Data]=(\n"
			 "[Size]=({:L})"
			 ")=[Data]\n",
			 duration_cast<std::chrono::nanoseconds>(vTimeTaken).count(),
			 duration_cast<std::chrono::microseconds>(vTimeTaken).count(),
			 duration_cast<std::chrono::milliseconds>(vTimeTaken).count(),
			 duration_cast<std::chrono::seconds>(vTimeTaken).count(),
			 vDataPack.size()
		 );
	 }},
	{"tFileReaderPreallocDouble",
	 []()
	 {
		 auto vTimeSince = std::chrono::high_resolution_clock::now();
		 auto vDataPath	 = dPathToResource "/mnist-train-images.idx3-ubyte";
		 auto vDataFile0 = nFileSystem::ifstream(vDataPath, std::ios::binary);
		 auto vDataFile1 = nFileSystem::ifstream(vDataPath, std::ios::binary);
		 fThrowIfNot(
			 vDataFile0.is_open() && vDataFile1.is_open(),
			 std::logic_error(
				 nTextFormat::format("failed to load the file: {}", vDataPath)
			 )
		 );
		 size_t vDataSize = vDataFile0.seekg(0, std::ios::end).tellg();
		 auto		vDataPack = std::vector<uint8_t>(vDataSize);
		 size_t vDataHalf = vDataSize >> 1;
		 vDataFile0.seekg(0, std::ios::beg);
		 vDataFile1.seekg(vDataHalf, std::ios::beg);
		 std::thread vFlow1(
			 [&]()
			 {
				 for(size_t vI = 0; vI < vDataHalf; vI++)
				 {
					 vDataFile0 >> vDataPack[vI];
				 }
			 }
		 );
		 std::thread vFlow2(
			 [&]()
			 {
				 for(size_t vI = vDataHalf; !vDataFile1.eof(); vI++)
				 {
					 vDataFile1 >> vDataPack[vI];
				 }
			 }
		 );
		 vFlow1.join();
		 vFlow2.join();
		 auto vTimeUntil = std::chrono::high_resolution_clock::now();
		 auto vTimeTaken = vTimeUntil - vTimeSince;
		 nTextFormat::print(
			 stdout,
			 "[TimeTaken]=(\n"
			 "[nanos]=({:L})"
			 "[micro]=({:L})"
			 "[milli]=({:L})"
			 "[secon]=({:L})"
			 ")=[TimeTaken]\n"
			 "[Data]=(\n"
			 "[Size]=({:L})"
			 ")=[Data]\n",
			 duration_cast<std::chrono::nanoseconds>(vTimeTaken).count(),
			 duration_cast<std::chrono::microseconds>(vTimeTaken).count(),
			 duration_cast<std::chrono::milliseconds>(vTimeTaken).count(),
			 duration_cast<std::chrono::seconds>(vTimeTaken).count(),
			 vDataPack.size()
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
		 auto vGraphOfNetwork
			 = tMakerOfNetwork()
					 .fMakeLayer<tLayerOfNetworkDense>(0x40, 0x30)
					 .fMakeLayer<tLayerOfNetworkActivSigma>(0x30)
					 .fMakeLayer<tLayerOfNetworkDense>(0x30, 0x20)
					 .fMakeLayer<tLayerOfNetworkActivTanh>(0x30)
					 .fMakeLayer<tLayerOfNetworkDense>(0x20, 0x10)
					 .fMakeLayer<tLayerOfNetworkActivRelu>(0x10)
					 .fTakeGraph();
		 std::clog << "[vGraphOfNetwork]=(" << std::endl;
		 std::clog << *vGraphOfNetwork << ")" << std::endl;
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
	{"tAiXorSolver",
	 []()
	 {
		 auto vGraphOfNetwork
			 = tMakerOfNetwork()
					 .fMakeLayer<tLayerOfNetworkDense>(2, 3)
					 .fMakeLayer<tLayerOfNetworkActivTanh>(3)
					 .fMakeLayer<tLayerOfNetworkDense>(3, 1)
					 .fMakeLayer<tLayerOfNetworkActivTanh>(1)
					 .fTakeGraph();
		 for(size_t vIndex = 1; vIndex <= 10'000; vIndex++)
		 {
			 auto vInputL = static_cast<bool>(vRandBool(vRandEngine));
			 auto vInputR = static_cast<bool>(vRandBool(vRandEngine));
			 auto vInputV = tVec(2);
			 vInputV[0]		= static_cast<tNum>(vInputL);
			 vInputV[1]		= static_cast<tNum>(vInputR);
			 auto vAnswer = tVec(1);
			 vAnswer[0]		= static_cast<tNum>(vInputL ^ vInputR);
			 vGraphOfNetwork->fLearn(vInputV, vAnswer);
		 }
		 for(auto vIndex = 0b000; vIndex < 0b100; vIndex++)
		 {
			 bool vInputL = (vIndex & 0b10) >> 1;
			 bool vInputR = (vIndex & 0b01) >> 0;
			 auto vInputV = tVec(2);
			 vInputV[0]		= static_cast<tNum>(vInputL);
			 vInputV[1]		= static_cast<tNum>(vInputR);
			 vGraphOfNetwork->fAhead(vInputV);
			 vInputV[0] = vInputV[0] > 0.5 ? 1.0 : 0.0;
			 nTextFormat::println("[{:d}^{:d}]={}", vInputL, vInputR, vInputV[0]);
		 }
	 }}, //tAiXorSolver
	{"tAiDigitReader",
	 []()
	 {
		 //timer
		 auto vTimeSince = std::chrono::high_resolution_clock::now();
		 //graph
		 auto vGraphOfNetwork
			 = tMakerOfNetwork()
					 .fMakeLayer<tLayerOfNetworkDense>(28 * 28, 32)
					 .fMakeLayer<tLayerOfNetworkActivTanh>(32)
					 .fMakeLayer<tLayerOfNetworkDense>(32, 16)
					 .fMakeLayer<tLayerOfNetworkActivTanh>(16)
					 .fMakeLayer<tLayerOfNetworkDense>(16, 10)
					 .fMakeLayer<tLayerOfNetworkActivTanh>(10)
					 .fTakeGraph();
		 try//learn
		 {
			 //image
			 auto vLearnImageFile = nFileSystem::ifstream(
				 dPathToResource "/mnist-train-images.idx3-ubyte", std::ios::binary
			 );
			 fThrowIfNot(
				 vLearnImageFile.is_open(),
				 std::logic_error("failed to load the mnist-train-images file")
			 );
			 //-//space
			 vLearnImageFile.seekg(0, std::ios::end);
			 size_t vLearnImageSpace = vLearnImageFile.tellg();
			 nTextFormat::println(stdout, "[LearnImageSpace]={}", vLearnImageSpace);
			 vLearnImageFile.seekg(4, std::ios::beg);
			 //-//count
			 auto vLearnImageCount = fRead<int32_t>(vLearnImageFile);
			 //-//sizes
			 nTextFormat::println(stdout, "[LearnImageCount]={}", vLearnImageCount);
			 auto vLearnImageSizeX = fRead<int32_t>(vLearnImageFile);//rows
			 nTextFormat::println(stdout, "[LearnImageSizeX]={}", vLearnImageSizeX);
			 auto vLearnImageSizeY = fRead<int32_t>(vLearnImageFile);//cols
			 nTextFormat::println(stdout, "[LearnImageSizeY]={}", vLearnImageSizeY);
			 //label
			 auto vLearnLabelFile = nFileSystem::ifstream(
				 dPathToResource "/mnist-train-labels.idx1-ubyte", std::ios::binary
			 );
			 fThrowIfNot(
				 vLearnLabelFile.is_open(),
				 std::logic_error("failed to load the mnist-train-labels file")
			 );
			 //-//space
			 vLearnLabelFile.seekg(0, std::ios::end);
			 size_t vLearnLabelSpace = vLearnLabelFile.tellg();
			 nTextFormat::println(stdout, "[LearnLabelsSpace]={}", vLearnLabelSpace);
			 vLearnLabelFile.seekg(4, std::ios::beg);
			 //-//count
			 auto vLearnLabelCount = fRead<int32_t>(vLearnLabelFile);
			 nTextFormat::println(stdout, "[LearnLabelCount]={}", vLearnLabelCount);
			 //learning
			 auto vTruth = tVec(10);					 //expected answer
			 auto vError = tVec(vTruth.size());//cost from each invididual example
			 auto vBatch = 10;								 //cost from a batch of examples
			 //process
			 for(size_t vIndex = 1;
					 (vIndex <= vLearnImageCount && vIndex <= vLearnLabelCount)
					 && (!vLearnImageFile.eof() && !vLearnLabelFile.eof());
					 vIndex++)
			 {
				 auto vInput = tVec(vLearnImageSizeX * vLearnImageSizeY);
				 for(size_t vY = 0; vY < vLearnImageSizeY; vY++)
				 {
					 for(size_t vX = 0; vX < vLearnImageSizeX; vX++)
					 {
						 auto vPixel		= fRead<uint8_t>(vLearnImageFile);
						 auto vIndex		= vY * vLearnImageSizeX + vX;
						 vInput[vIndex] = static_cast<tNum>(vPixel) / 255.0;
					 }
				 }
				 std::fill(vTruth.begin(), vTruth.end(), 0.0);
				 auto vDigit		= fRead<uint8_t>(vLearnLabelFile);
				 vTruth[vDigit] = 1.0;
#if 0
				 vGraphOfNetwork->fAhead(vInput);
				 vError
					 = vError + (vInput - vTruth) * 2.0 / static_cast<tNum>(vTruth.size());
				 if(vIndex % vBatch == 0)
				 {
					 vError = vError / static_cast<tNum>(vBatch);
					 vGraphOfNetwork->fAback(vError);
					 vError = tVec(vTruth.size());
				 }
#else
				 vGraphOfNetwork->fLearn(vInput, vTruth);
#endif
				 auto vShowIndex = 10'000;
				 if(vIndex % vShowIndex == 0)
				 {
					 nTextFormat::println(
						 stdout,
						 "[Learn]{}/{}",
						 vIndex / vShowIndex,
						 vLearnLabelCount / vShowIndex
					 );
				 }
			 }
		 }//learn
		 catch(std::exception &vError)
		 {
			 nTextFormat::println(stderr, "failed learn process: {}", vError.what());
			 return;
		 }
		 try//trial
		 {
			 //image
			 auto vTrialImageFile = nFileSystem::ifstream(
				 dPathToResource "/mnist-t10k-images.idx3-ubyte", std::ios::binary
			 );
			 fThrowIfNot(
				 vTrialImageFile.is_open(),
				 std::logic_error("failed to load the mnist-t10k-images file")
			 );
			 //-//space
			 vTrialImageFile.seekg(0, std::ios::end);
			 size_t vTrialImageSpace = vTrialImageFile.tellg();
			 nTextFormat::println(stdout, "[TrialImageSpace]={}", vTrialImageSpace);
			 vTrialImageFile.seekg(4, std::ios::beg);
			 //-//count
			 auto vTrialImageCount = fRead<int32_t>(vTrialImageFile);
			 nTextFormat::println("[TrialImageCount]={}", vTrialImageCount);
			 //-//sizes
			 auto vTrialImageSizeX = fRead<int32_t>(vTrialImageFile);//rows
			 nTextFormat::println("[TrialImageSizeX]={}", vTrialImageSizeX);
			 auto vTrialImageSizeY = fRead<int32_t>(vTrialImageFile);//cols
			 nTextFormat::println("[TrialImageSizeY]={}", vTrialImageSizeY);
			 //label
			 auto vTrialLabelFile = nFileSystem::ifstream(
				 dPathToResource "/mnist-t10k-labels.idx1-ubyte", std::ios::binary
			 );
			 fThrowIfNot(
				 vTrialLabelFile.is_open(),
				 std::logic_error("failed to load the mnist-t10k-labels file")
			 );
			 //-//space
			 vTrialLabelFile.seekg(0, std::ios::end);
			 size_t vTrialLabelSpace = vTrialLabelFile.tellg();
			 nTextFormat::println(stdout, "[TrialLabelSpace]={}", vTrialLabelSpace);
			 vTrialLabelFile.seekg(4, std::ios::beg);
			 //-//count
			 auto vTrialLabelCount = fRead<int32_t>(vTrialLabelFile);
			 nTextFormat::println(stdout, "[TrialLabelCount]={}", vTrialLabelCount);
			 //process
			 for(size_t vIndex = 1;
					 (vIndex <= vTrialImageCount && vIndex <= vTrialLabelCount)
					 && (!vTrialImageFile.eof() && !vTrialLabelFile.eof());
					 vIndex++)
			 {
				 auto vInput = tVec(vTrialImageSizeX * vTrialImageSizeY);
				 for(size_t vY = 0; vY < vTrialImageSizeY; vY++)
				 {
					 for(size_t vX = 0; vX < vTrialImageSizeX; vX++)
					 {
						 auto vPixel		= fRead<uint8_t>(vTrialImageFile);
						 auto vIndex		= vY * vTrialImageSizeX + vX;
						 vInput[vIndex] = static_cast<tNum>(vPixel) / 255.0;
					 }
				 }
				 auto vLabel		= fRead<uint8_t>(vTrialLabelFile);
				 auto vTruth		= tVec(10);
				 vTruth[vLabel] = 1.0;
				 vGraphOfNetwork->fLearn(vInput, vTruth);
				 if(vIndex % 1'000 == 0)
				 {
					 auto vTruthIndex = std::max_element(vTruth.begin(), vTruth.end());
					 auto vTruthDigit = vTruthIndex - vTruth.begin();
					 auto vInputIndex = std::max_element(vInput.begin(), vInput.end());
					 auto vInputDigit = vInputIndex - vInput.begin();
					 nTextFormat::println(
						 stdout, "[Output] = {} [Answer] = {}", vInputDigit, vTruthDigit
					 );
				 }
			 }
		 }//trial
		 catch(std::exception &vError)
		 {
			 nTextFormat::println(stderr, "failed trial process: {}", vError.what());
			 return;
		 }
		 //timer
		 auto vTimeUntil = std::chrono::high_resolution_clock::now();
		 nTextFormat::println(
			 "[TimeTaken][milli]={}",
			 duration_cast<std::chrono::milliseconds>(vTimeUntil - vTimeSince).count()
		 );
	 }}, //tAiDigitReader
};
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
		 static_cast<tNum>(vWindow.getSize().x),
		 static_cast<tNum>(vWindow.getSize().y),
	 };
	sf::Vector2f vWindowSizeHalf;
	vWindowSizeHalf.x = static_cast<tNum>(vWindowSizeFull.x) / 2.0;
	vWindowSizeHalf.y = static_cast<tNum>(vWindowSizeFull.y) / 2.0;
	//timing
	sf::Clock vClock;
	sf::Time	vTimePNow = vClock.getElapsedTime();
	sf::Time	vTimePWas = vTimePNow;
	tNum			vTimeFNow = vTimePWas.asSeconds();
	tNum			vTimeFWas = vTimeFNow;
	unsigned	vTimeIWas = static_cast<unsigned>(vTimeFWas);
	unsigned	vTimeINow = static_cast<unsigned>(vTimeFNow);
	//intel
	auto vGraphOfNetwork
		= tMakerOfNetwork()
				.fMakeLayer<tLayerOfNetworkDense>(28 * 28, 32)
				.fMakeLayer<tLayerOfNetworkActivTanh>(32)
				.fMakeLayer<tLayerOfNetworkDense>(32, 16)
				.fMakeLayer<tLayerOfNetworkActivTanh>(16)
				.fMakeLayer<tLayerOfNetworkDense>(16, 10)
				.fMakeLayer<tLayerOfNetworkActivTanh>(10)
				.fTakeGraph();
	//-//image
	auto vImageFile = nFileSystem::ifstream(
		dPathToResource "/mnist-train-images.idx3-ubyte", std::ios::binary
	);
	fThrowIfNot(
		vImageFile.is_open(),
		std::logic_error("failed to load the mnist-train-images file")
	);
	//-//-//space
	vImageFile.seekg(0, std::ios::end);
	size_t vImageSpace = vImageFile.tellg();
	nTextFormat::println(stdout, "[ImageSpace]={}", vImageSpace);
	vImageFile.seekg(4, std::ios::beg);
	//-//-//count
	auto vImageCount = fRead<int32_t>(vImageFile);
	//-//-//sizes
	nTextFormat::println(stdout, "[ImageCount]={}", vImageCount);
	auto vImageSizeX = fRead<int32_t>(vImageFile);//rows
	nTextFormat::println(stdout, "[ImageSizeX]={}", vImageSizeX);
	auto vImageSizeY = fRead<int32_t>(vImageFile);//cols
	nTextFormat::println(stdout, "[ImageSizeY]={}", vImageSizeY);
	auto vImageIndex = vImageCount - vImageCount;
	//-//label
	auto vLabelFile = nFileSystem::ifstream(
		dPathToResource "/mnist-train-labels.idx1-ubyte", std::ios::binary
	);
	fThrowIfNot(
		vLabelFile.is_open(),
		std::logic_error("failed to load the mnist-train-labels file")
	);
	//-//-//space
	vLabelFile.seekg(0, std::ios::end);
	size_t vLabelSpace = vLabelFile.tellg();
	nTextFormat::println(stdout, "[LabelsSpace]={}", vLabelSpace);
	vLabelFile.seekg(4, std::ios::beg);
	//-//-//count
	auto vLabelCount = fRead<int32_t>(vLabelFile);
	nTextFormat::println(stdout, "[LabelCount]={}", vLabelCount);
	auto vLabelIndex = vLabelCount - vLabelCount;
	//-//learn
	auto vTruth = tVec(10);						//expected answer
	auto vError = tVec(vTruth.size());//cost from each invididual example
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
		auto vImageReset = (vImageIndex > vImageCount) || vImageFile.eof();
		auto vLabelReset = (vLabelIndex > vLabelCount) || vLabelFile.eof();
		if(vImageReset)
		{
			vImageFile.seekg(0, std::ios::beg);
			vImageIndex = 0;
		}
		else
		{
			vImageIndex++;
		}
		if(vLabelReset)
		{
			vLabelFile.seekg(0, std::ios::beg);
			vLabelIndex = 0;
		}
		else
		{
			vLabelIndex++;
		}
		if(vImageReset || vLabelReset)
		{
			continue;
		}
		auto vInput = tVec(vImageSizeX * vImageSizeY);
		for(size_t vY = 0; vY < vImageSizeY; vY++)
		{
			for(size_t vX = 0; vX < vImageSizeX; vX++)
			{
				auto vPixel		 = fRead<uint8_t>(vImageFile);
				auto vIndex		 = vY * vImageSizeX + vX;
				vInput[vIndex] = static_cast<tNum>(vPixel) / 255.0;
			}
		}
		std::fill(vTruth.begin(), vTruth.end(), 0.0);
		auto vDigit		 = fRead<uint8_t>(vLabelFile);
		vTruth[vDigit] = 1.0;
		vGraphOfNetwork->fLearn(vInput, vTruth);
#if 1
		auto vShowIndex = 10'000;
		if(vLabelIndex % vShowIndex == 0)
		{
			auto vPercent = vLabelIndex / vShowIndex;
			auto vTotal		= vLabelCount / vShowIndex;
			nTextFormat::println(stdout, "[Trial]{}/{}", vPercent, vTotal);
		}
#endif
		sf::Transform vTform{sf::Transform::Identity};
		vTform.translate(sf::Vector2f{
			static_cast<tNum>(vWindow.getSize().x) * +0.1f,
			static_cast<tNum>(vWindow.getSize().y) * +0.1f,
		});
		vTform.scale(sf::Vector2f{
			static_cast<tNum>(vWindow.getSize().x) * +0.8f,
			static_cast<tNum>(vWindow.getSize().y) * +0.8f,
		});
		vWindow.clear();
		vGraphOfNetwork->fDraw(vWindow, vTform);
		vWindow.display();
		//events
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
