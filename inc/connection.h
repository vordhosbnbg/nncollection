#pragma once
#include <cstddef>

class Neuron;

struct Connection
{
    Connection() : inputWeight(0.0) {}
    Connection( float weight): inputWeight(weight) {}

    template<typename Archive>
    void load(Archive& archive)
    {
        archive.load("inputWeight", inputWeight);
    }

    template<typename Archive>
    void save(Archive& archive) const
    {
        archive.save("inputWeight", inputWeight);
    }


    float inputWeight;
};
