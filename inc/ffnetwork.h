#pragma once

#include <array>
#include <tuple>

#include <layer.h>

template <unsigned int inputNb, unsigned int outputNb, unsigned int... hiddenNb>
class FFNetwork
{
public:
    constexpr FFNetwork()
    {
        connectLayers();
    }
    ~FFNetwork() = default;



private:
    constexpr inline void connectLayers()
    {
        constexpr size_t hiddenLayersCount = std::tuple_size<decltype(hiddenLayers)>::value;
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
        for(size_t hiddenLayerNb = 0; hiddenLayerNb < hiddenLayersCount ; ++hiddenLayerNb)
        {

        }
    }

    Layer<inputNb> inputLayer;
    Layer<outputNb> outputLayer;
    std::tuple<Layer<hiddenNb>...> hiddenLayers;
};
