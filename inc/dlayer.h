#pragma once
#include <vector>
#include <boost/align.hpp>
#include <cmath>
#include <algorithm>
#include <random>
#include <execution>
#include "precomputedtanh.h"

using AlignedFloatVector = std::vector<float, boost::alignment::aligned_allocator<float,16>>;
static constexpr PrecomputedTanh<10000,-10,10> prec = PrecomputedTanh<10000,-10,10>();

struct DynamicLayer
{
    void connect(const DynamicLayer& inputLayer, bool ensureProperConnections = false)
    {
        prevLayer = &inputLayer;
        if(ensureProperConnections)
        {
            for(size_t neuronIdx = 0; neuronIdx < neurons.size(); ++neuronIdx)
            {
                AlignedFloatVector& weights = weightsPerNeuron[neuronIdx];
                weights.resize(inputLayer.size(), 0.0);
            }
        }
    }

    size_t size() const
    {
        return neurons.size();
    }

    size_t inputsNb() const
    {
        if(prevLayer)
        {
            return prevLayer->size();
        }

        return 0;
    }

    void setNeuronNb(size_t newSize)
    {
        neurons.resize(newSize, 0.0);
        AlignedFloatVector weightsForNewNeuron(inputsNb(), 0.0);

        weightsPerNeuron.resize(newSize, weightsForNewNeuron);
    }

    size_t getNeuronNb()
    {
        return neurons.size();
    }

    void setNeuronValue(size_t inputId, float value)
    {
        neurons[inputId] = value;
    }

    float getNeuronValue(size_t inputId) const
    {
        return neurons[inputId];
    }

    void addNewInput()
    {
        // add weights for each value
        for(size_t neuronIdx = 0; neuronIdx < neurons.size(); ++neuronIdx)
        {
            AlignedFloatVector& weights = weightsPerNeuron[neuronIdx];
            weights.emplace_back(0.0);
        }
    }

    void removeInput(size_t idx)
    {
        for(size_t neuronIdx = 0; neuronIdx < neurons.size(); ++neuronIdx)
        {
            weightsPerNeuron[neuronIdx].erase(weightsPerNeuron[neuronIdx].begin() + idx);
        }
    }

    void update()
    {
        if(prevLayer)
        {
            const AlignedFloatVector& inputs = prevLayer->neurons;
            for(size_t neuronIdx = 0; neuronIdx < neurons.size(); ++neuronIdx)
            {
                float& neuron = neurons[neuronIdx];
                const AlignedFloatVector& weights = weightsPerNeuron[neuronIdx];

                //neuron = std::inner_product(inputs.begin(), inputs.end(), weights.begin(), 0.0f);
                neuron = std::transform_reduce(std::execution::unseq, inputs.begin(), inputs.end(), weights.begin(), 0.0f);
                neuron += bias;
                neuron = prec.tanh(neuron);
            }
        }
    }

    void randomizeInitial(std::mt19937& randE,
                          std::uniform_real_distribution<float>& biasDist,
                          std::uniform_real_distribution<float>& weightDist)
    {
        bias = biasDist(randE);
        for (AlignedFloatVector& weights : weightsPerNeuron)
        {
            for(float& weight : weights)
            {
                weight = weightDist(randE);
            }
        }
    }

    void mutate(std::mt19937& randE,
                std::uniform_real_distribution<float>& positiveNormalizedDist,
                float biasMutChance,
                std::normal_distribution<float>& biasMutRate,
                float weightMutChance,
                std::normal_distribution<float>& weightMutRate)
    {
        float actualMutBias = positiveNormalizedDist(randE);
        if(actualMutBias < biasMutChance)
        {
            float biasChange = biasMutRate(randE);
            bias += biasChange;
            if(bias > 1)
            {
                bias = 1;
            }
            else if(bias < -1)
            {
                bias = -1;
            }
        }
        for (AlignedFloatVector& weights : weightsPerNeuron)
        {
            for(float& weight : weights)
            {
                float actualMutWeight = positiveNormalizedDist(randE);
                if(actualMutWeight < weightMutChance)
                {
                    float weightChange = weightMutRate(randE);
                    weight += weightChange;
                    std::clamp(weight, -1.0f, 1.0f);
                }
            }
        }

    }

    template<typename Archive>
    void load(Archive& archive)
    {
        archive.load("neurons", neurons);
        archive.load("weightsPerNeuron", weightsPerNeuron);
        archive.load("bias", bias);
    }

    template<typename Archive>
    void save(Archive& archive) const
    {
        archive.save("neurons", neurons);
        archive.save("weightsPerNeuron", weightsPerNeuron);
        archive.save("bias", bias);
    }

private:
    AlignedFloatVector neurons;
    std::vector<AlignedFloatVector> weightsPerNeuron;
    float bias = 0.0;
    const DynamicLayer* prevLayer = nullptr;
};
