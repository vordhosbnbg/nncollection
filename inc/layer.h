#pragma once

#include <array>
#include <random>

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

    void randomizeInitial(std::mt19937& randE,
                                 std::uniform_real_distribution<float>& biasDist,
                                 std::uniform_real_distribution<float>& weightDist)
    {
        for(Neuron& neuron : neurons)
        {
            neuron.randomizeInitial(randE,
                                    biasDist,
                                    weightDist);
        }
    }

    void mutate(std::mt19937& randE,
                       std::uniform_real_distribution<float>& positiveNormalizedDist,
                       float biasMutChance,
                       std::uniform_real_distribution<float>& biasMutRate,
                       float weightMutChance,
                       std::uniform_real_distribution<float>& weightMutRate)
    {
        for(Neuron& neuron : neurons)
        {
            neuron.mutate(randE,
                          positiveNormalizedDist,
                          biasMutChance,
                          biasMutRate,
                          weightMutChance,
                          weightMutRate);
        }
    }

    constexpr void update()
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

