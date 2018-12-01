#pragma once

#include <array>
#include <tuple>
#include <random>
#include "layer.h"

template <unsigned int inputNb, unsigned int outputNb, unsigned int... hiddenNb>
class FFNetwork
{
public:
    constexpr FFNetwork()
    {
        connectLayers();
        randomizeInitial(re,
                         _biasDist,
                         _weightDist);
    }
    ~FFNetwork() = default;


    constexpr inline void process()
    {
        processHiddenLayerRecurse<0>();
        outputLayer.update();
    }


private:
    std::random_device rd;
    std::mt19937 re{rd()};
    Layer<inputNb> inputLayer;
    Layer<outputNb> outputLayer;
    std::tuple<Layer<hiddenNb>...> hiddenLayers;
    static constexpr size_t hiddenLayersCount = std::tuple_size<decltype(hiddenLayers)>::value;
    std::uniform_real_distribution<float> _biasDist{-1,1};
    std::uniform_real_distribution<float> _weightDist{-1,1};


    template<size_t layerNb>
    typename std::enable_if<layerNb == hiddenLayersCount>::type
    inline randomizeLayerInitialAndRecurse([[maybe_unused]] std::mt19937& randE,
                                           [[maybe_unused]] std::uniform_real_distribution<float>& biasDist,
                                           [[maybe_unused]] std::uniform_real_distribution<float>& weightDist)
    {
    }

    template<size_t layerNb>
    typename std::enable_if<layerNb < hiddenLayersCount>::type
    inline randomizeLayerInitialAndRecurse(std::mt19937& randE,
                                           std::uniform_real_distribution<float>& biasDist,
                                           std::uniform_real_distribution<float>& weightDist)
    {
        auto& hiddenLayerCurrent = std::get<layerNb>(hiddenLayers);
        hiddenLayerCurrent.randomizeInitial(randE,
                                            biasDist,
                                            weightDist);
        randomizeLayerInitialAndRecurse<layerNb+1>(randE,
                                                   biasDist,
                                                   weightDist);
    }

    inline void randomizeInitial(std::mt19937& randE,
                                 std::uniform_real_distribution<float>& biasDist,
                                 std::uniform_real_distribution<float>& weightDist)
    {
        randomizeLayerInitialAndRecurse<0>(randE,
                                           biasDist,
                                           weightDist);
        outputLayer.randomizeInitial(randE,
                                     biasDist,
                                     weightDist);
    }

    template<size_t layerNb>
    typename std::enable_if<layerNb <= 1>::type
    constexpr inline connectNthHiddenLayerToPreviousAndRecurse()
    {
    }

    template<size_t layerNb>
    typename std::enable_if<layerNb >= 2>::type
    constexpr inline connectNthHiddenLayerToPreviousAndRecurse()
    {
        auto& hiddenLayerPrevious = std::get<layerNb-2>(hiddenLayers);
        auto& hiddenLayerCurrent = std::get<layerNb-1>(hiddenLayers);
        hiddenLayerCurrent.connectInputsFrom(hiddenLayerPrevious);
        connectNthHiddenLayerToPreviousAndRecurse<layerNb-1>();
    }
    constexpr inline void connectLayers()
    {
        if(hiddenLayersCount == 0)
        {
            outputLayer.connectInputsFrom(inputLayer);
        }
        else
        {
            auto& hiddenLayerFirst = std::get<0>(hiddenLayers);
            auto& hiddenLayerLast = std::get<hiddenLayersCount-1>(hiddenLayers);
            hiddenLayerFirst.connectInputsFrom(inputLayer);
            outputLayer.connectInputsFrom(hiddenLayerLast);
        }
        if(hiddenLayersCount > 1)
        {
            connectNthHiddenLayerToPreviousAndRecurse<hiddenLayersCount>();
        }
    }

    template<size_t layerNb>
    typename std::enable_if<layerNb == hiddenLayersCount>::type
    constexpr inline processHiddenLayerRecurse()
    {
    }

    template<size_t layerNb>
    typename std::enable_if<layerNb < hiddenLayersCount>::type
    constexpr inline processHiddenLayerRecurse()
    {
        auto& hiddenLayerCurrent = std::get<layerNb>(hiddenLayers);
        hiddenLayerCurrent.update();
        processHiddenLayerRecurse<layerNb+1>();
    }

};
