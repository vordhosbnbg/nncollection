#pragma once
#include <random>
#include "dlayer.h"

struct DynamicNetwork
{

    void connectNetwork()
    {
        if(hiddenLayers.size() > 0)
        {
            hiddenLayers[0].connect(inputLayer);
            for(size_t layerIdx = 1; layerIdx < hiddenLayers.size(); ++layerIdx)
            {
                DynamicLayer& prevLayer = hiddenLayers[layerIdx-1];
                DynamicLayer& currentLayer = hiddenLayers[layerIdx];
                currentLayer.connect(prevLayer);
            }
            outputLayer.connect(hiddenLayers.back());
        }
    }

    void process()
    {
        for(DynamicLayer& hiddenLayer : hiddenLayers)
        {
            hiddenLayer.update();
        }
        outputLayer.update();
    }


    template<typename ...Ints>
    void setHiddenLayers(Ints... ints)
    {
        size_t neuronsNb[] = {ints...};
        for(size_t neuronsForLayer : neuronsNb)
        {
            DynamicLayer& newHiddenLayer = hiddenLayers.emplace_back();
            newHiddenLayer.setNeuronNb(neuronsForLayer);
        }

    }

    void setInputNb(size_t inputNb)
    {
        inputLayer.setNeuronNb(inputNb);
        if(hiddenLayers.size() > 0)
        {
            if(hiddenLayers[0].inputsNb() > inputNb)
            {
                size_t inputsToRemove = hiddenLayers[0].inputsNb() - inputNb;

                while(inputsToRemove)
                {
                    hiddenLayers[0].removeInput(hiddenLayers[0].inputsNb() - inputNb + inputsToRemove);
                    --inputsToRemove;
                }
            }
            else if(hiddenLayers[0].inputsNb() < inputNb)
            {
                size_t inputsToAdd = inputNb - hiddenLayers[0].inputsNb();

                while(inputsToAdd)
                {
                    hiddenLayers[0].addNewInput();
                    --inputsToAdd;
                }
            }
        }
    }

    void setOutputNb(size_t outputNb)
    {
        outputLayer.setNeuronNb(outputNb);
    }

    void setInputValue(size_t inputId, float value)
    {
        inputLayer.setNeuronValue(inputId, value);
    }

    float getOutputValue(size_t outputId)
    {
        return outputLayer.getNeuronValue(outputId);
    }


    template<typename Archive>
    void load(Archive& archive)
    {
        archive.load("inputLayer", inputLayer);
        archive.load("hiddenLayers", hiddenLayers);
        archive.load("outputLayer", outputLayer);
    }

    template<typename Archive>
    void save(Archive& archive) const
    {
        archive.save("inputLayer", inputLayer);
        archive.save("hiddenLayers", hiddenLayers);
        archive.save("outputLayer", outputLayer);
    }

private:
    const std::uniform_real_distribution<float> normalizedDist{-1,1};
    const std::uniform_real_distribution<float> positiveNormalizedDist{0,1};
    DynamicLayer inputLayer;
    std::vector<DynamicLayer> hiddenLayers;
    DynamicLayer outputLayer;
};
