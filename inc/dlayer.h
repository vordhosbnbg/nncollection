#pragma once
#include <vector>
#include <boost/align.hpp>
#include <cmath>
#include <algorithm>
#include "precomputedtanh.h"

using AlignedFloatVector = std::vector<float, boost::alignment::aligned_allocator<float,16>>;

struct DynamicLayer
{
    void connect(const DynamicLayer& inputLayer)
    {
        prevLayer = &inputLayer;
        for(size_t neuronIdx = 0; neuronIdx < neurons.size(); ++neuronIdx)
        {
            AlignedFloatVector& weights = weightsPerNeuron[neuronIdx];
            weights.resize(inputLayer.size(), 0.0);
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

    void setNeuronValue(size_t inputId, float value)
    {
        neurons[inputId] = value;
    }

    float getNeuronValue(size_t inputId)
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
            size_t inputSize = inputsNb();
            const AlignedFloatVector& inputs = prevLayer->neurons;
            for(size_t neuronIdx = 0; neuronIdx < neurons.size(); ++neuronIdx)
            {
                float& neuron = neurons[neuronIdx];
                const AlignedFloatVector& weights = weightsPerNeuron[neuronIdx];

                neuron = 0;
                for(size_t inputIdx = 0; inputIdx < inputSize; ++inputIdx)
                {
                    neuron += inputs[inputIdx] * weights[inputIdx];
                }
                neuron += bias;
                neuron = /*prec.*/std::tanh(neuron);
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

    //static constexpr PrecomputedTanh<10000,-10,10> prec = PrecomputedTanh<10000,-10,10>();
};
