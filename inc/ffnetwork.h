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
    }
    ~FFNetwork() = default;


    constexpr inline void process()
    {
        processHiddenLayerRecurse<0>();
        outputLayer.update();
    }


private:
    std::random_device rd;
    std::mt19937 re{rd()}; // or std::default_random_engine e{rd()};

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


    Layer<inputNb> inputLayer;
    Layer<outputNb> outputLayer;
    std::tuple<Layer<hiddenNb>...> hiddenLayers;
    static constexpr size_t hiddenLayersCount = std::tuple_size<decltype(hiddenLayers)>::value;

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
