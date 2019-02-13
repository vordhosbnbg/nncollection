#pragma once

#include <array>
#include <random>

#include "neuron.h"

template <size_t Nb>
struct Layer
{

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

    template<size_t PrevLayerNeurons>
    constexpr void update(const Layer<PrevLayerNeurons>& prevLayer)
    {
        for(Neuron& neuron : neurons)
        {
            neuron.updateInit();
            for(size_t inputIdx = 0; inputIdx < prevLayer.getNeuronNb(); ++inputIdx)
            {
                neuron.updateFromInput(inputIdx, prevLayer.neurons[inputIdx]);
            }
            neuron.updateEnd();
        }
    }

    template<size_t PrevLayerNeurons>
    constexpr void addInputs([[maybe_unused]] const Layer<PrevLayerNeurons>& prevLayer)
    {
        for(Neuron& neuron : neurons)
        {
            neuron.resizeInputs(PrevLayerNeurons);
        }
    }

    constexpr unsigned int getNeuronNb() const
    {
        return Nb;
    }

    std::array<Neuron, Nb> neurons;
};

