#pragma once
#include <vector>

#include "connection.h"

class Neuron
{
public:
    Neuron() : value(0.0){}
    ~Neuron() = default;

    inline void addInput(const Neuron& inputNeuron, double weight)
    {
        inputs.emplace_back(inputNeuron, weight);
    }

    inline double getValue() const
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
        value /= inputs.size();
    }

private:

    double value;
    std::vector<Connection> inputs;

};
