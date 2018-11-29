#pragma once
#include <vector>
#include <cmath>

#include "connection.h"

class Neuron
{
public:
    Neuron() : value(0), bias(0) {}
    ~Neuron() = default;

    inline void addInput(const Neuron& inputNeuron, float weight)
    {
        inputs.emplace_back(inputNeuron, weight);
    }

    inline float getValue() const
    {
        return value;
    }

    inline void update()
    {
        value = 0.0;
        for(const Connection& input : inputs)
        {
            value += input.inputNeuron.value * input.inputWeight;
        }
        value += bias;
        value = std::tanh(value);
    }

private:

    float value;
    float bias;
    std::vector<Connection> inputs;

};
