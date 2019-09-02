#pragma once
#include "dynamicnetwork.h"

template <unsigned int inputNb, unsigned int outputNb, unsigned int... hiddenNb>
struct DynamicNetworkAdapter : public DynamicNetwork
{
    DynamicNetworkAdapter(std::mt19937& randomEngine) : DynamicNetwork(randomEngine)
    {
        setInputNb(inputNb);
        setOutputNb(outputNb);
        size_t neuronsNb[] = {hiddenNb...};

        for(size_t neuronsForLayer : neuronsNb)
        {
            addHiddenLayer(neuronsForLayer);
        }

        connectNetwork();
    }


    template<unsigned int outputId>
    float getOutput() const
    {
        static_assert (outputId < outputNb, "outputId must be less than outputNb");
        return getOutputValue(outputId);
    }

    template<unsigned int inputId>
    void setInput(float val)
    {
        static_assert (inputId < inputNb, "inputId must be less than inputNb");
        setInputValue(inputId, val);
    }


    static constexpr size_t getInputNb()
    {
        return inputNb;
    }

    static constexpr size_t getOutputNb()
    {
        return outputNb;
    }
};
