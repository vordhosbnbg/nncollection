#pragma once

class Neuron;

struct Connection
{
    Connection(const Neuron& neuron, float weight): inputNeuron(neuron), inputWeight(weight) {}
    ~Connection() = default;

    const Neuron& inputNeuron;
    float inputWeight;
};
