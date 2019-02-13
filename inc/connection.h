#pragma once
#include <cstddef>

class Neuron;

struct Connection
{
    Connection( float weight): inputWeight(weight) {}

    float inputWeight;
};
