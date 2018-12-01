#pragma once
#include <cmath>
#include <vector>

#include "connection.h"

class Neuron
{
public:
    Neuron() = default;
    Neuron(const Neuron& other) = default;
    Neuron& operator=(const Neuron& other) = default;
    Neuron(Neuron&& other) noexcept= default;
    Neuron& operator=(Neuron&& other) noexcept= default;
    ~Neuron() = default;

    inline void addInput(const Neuron& inputNeuron, float weight)
    {
        inputs.emplace_back(inputNeuron, weight);
    }

    inline void reserveInputs(size_t size)
    {
        inputs.reserve(size);
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

    float value{0};
    float bias{0};
    std::vector<Connection> inputs;

};
