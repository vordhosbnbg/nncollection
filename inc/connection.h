#pragma once

class Neuron;

struct Connection
{
    Connection(const Neuron& neuron, double weight): inputNeuron(neuron), inputWeight(weight) {}
    ~Connection() = default;

    const Neuron& inputNeuron;
    double inputWeight;
};
