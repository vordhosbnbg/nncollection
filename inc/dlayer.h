#pragma once
#include <vector>
#include <boost/align.hpp>
#include <cmath>
#include "precomputedtanh.h"

using AlignedFloatVector = std::vector<float, boost::alignment::aligned_allocator<float,16>>;

struct DynamicLayer
{
    void connect(const DynamicLayer* inputLayer)
    {
        if(inputLayer)
        {
            prevLayer = inputLayer;
            weights.resize(inputLayer->size());
        }
    }

    size_t size() const
    {
        return values.size();
    }

    size_t inputsNb() const
    {
        if(prevLayer)
        {
            return prevLayer->size();
        }

        return 0;
    }

    void addNewInput()
    {
        // add weights for each value
        size_t newWeightSize = weights.size() + size();
        weights.reserve(newWeightSize);
        for(auto pos = weights.begin() + inputsNb(); pos < weights.begin() + newWeightSize; pos += size())
        {
            weights.insert(pos, 0.0);
        }
    }

    void removeInput(size_t idx)
    {
        size_t weightsIdx = idx;
        for(size_t valIdx = 0; valIdx < size(); ++valIdx)
        {
            weights.erase(weights.begin()+weightsIdx);
            weightsIdx += size()-1; // backtrack with 1 to compensate for removed element
        }
    }

    void update()
    {
        if(prevLayer)
        {
            size_t weightsIdx = 0;
            size_t inputSize = inputsNb();
            const AlignedFloatVector& inputs = prevLayer->values;
            for(float& value : values)
            {
                value = 0;

                for(size_t inputIdx = 0; inputIdx < inputSize; ++inputIdx)
                {
                    value += inputs[inputIdx] * weights[weightsIdx+inputIdx];
                }
                value += bias;
                value = prec.tanh(value);
                weightsIdx += size();
            }
        }
    }

private:
    AlignedFloatVector values;
    AlignedFloatVector weights;
    float bias = 0;
    const DynamicLayer* prevLayer = nullptr;

    static constexpr PrecomputedTanh<10000,-10,10> prec = PrecomputedTanh<10000,-10,10>();
};
