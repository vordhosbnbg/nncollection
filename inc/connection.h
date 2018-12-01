#pragma once
#include <cstddef>

class Neuron;

struct Connection
{
    Connection(const Neuron& neuron, float weight): inputNeuron(neuron), inputWeight(weight) {}

    const Neuron& inputNeuron;
    float inputWeight;
};
