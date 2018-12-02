#pragma once
#include <cmath>
#include <vector>
#include <random>

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

    void addInput(const Neuron& inputNeuron, float weight)
    {
        inputs.emplace_back(inputNeuron, weight);
    }

    void reserveInputs(size_t size)
    {
        inputs.reserve(size);
    }

    float getValue() const
    {
        return value;
    }

    float setValue(float val)
    {
        return value = val;
    }

    void update()
    {
        value = 0.0;
        for(const Connection& input : inputs)
        {
            value += input.inputNeuron.value * input.inputWeight;
        }
        value += bias;
        value = std::tanh(value);
    }

    void mutate(std::mt19937& randE,
                       std::uniform_real_distribution<float>& positiveNormalizedDist,
                       float biasMutChance,
                       std::uniform_real_distribution<float>& biasMutRate,
                       float weightMutChance,
                       std::uniform_real_distribution<float>& weightMutRate)
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
        for(Connection& connection : inputs)
        {
            float actualMutWeight = positiveNormalizedDist(randE);
            if(actualMutWeight < weightMutChance)
            {
                float weightChange = weightMutRate(randE);
                connection.inputWeight += weightChange;
                if(connection.inputWeight > 1)
                {
                    connection.inputWeight = 1;
                }
                else if(connection.inputWeight < -1)
                {
                    connection.inputWeight = -1;
                }
            }
        }
    }

    void randomizeInitial(std::mt19937& randE,
                                 std::uniform_real_distribution<float>& biasDist,
                                 std::uniform_real_distribution<float>& weightDist)
    {
        bias = biasDist(randE);
        for(Connection& connection : inputs)
        {
            connection.inputWeight = weightDist(randE);
        }
    }

private:

    float value{0};
    float bias{0};
    std::vector<Connection> inputs;
};
