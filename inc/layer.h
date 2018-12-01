#pragma once

#include <array>

#include "neuron.h"

template <unsigned int Nb>
struct Layer
{
    template <unsigned int otherLayerNeuronNb>
    constexpr void connectInputsFrom(Layer<otherLayerNeuronNb>& otherLayer)
    {
        for(size_t cnt = 0; cnt < Nb; ++cnt)
        {
            neurons[cnt].reserveInputs(otherLayerNeuronNb);
            for(size_t otherCnt = 0; otherCnt < otherLayerNeuronNb; ++otherCnt)
            {
                neurons[cnt].addInput(otherLayer.neurons[otherCnt], 0.0);
            }
        }
    }

    constexpr inline void update()
    {
        for(Neuron& neuron : neurons)
        {
            neuron.update();
        }
    }

    constexpr unsigned int getNeuronNb() const
    {
        return Nb;
    }

    std::array<Neuron, Nb> neurons;
};

