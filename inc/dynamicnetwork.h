#pragma once
#include <random>
#include "dlayer.h"

struct DynamicNetwork
{

    DynamicNetwork(std::mt19937& randomEngine) : re(randomEngine)
    {
    }

    ~DynamicNetwork() = default;
    DynamicNetwork(const DynamicNetwork& other) :
        re(other.re),
        normalizedDist(other.normalizedDist),
        positiveNormalizedDist(other.positiveNormalizedDist),
        inputLayer(other.inputLayer),
        hiddenLayers(other.hiddenLayers),
        outputLayer(other.outputLayer)
    {
        reconstructLayerReferences();
    }

    DynamicNetwork(const DynamicNetwork&& other) :
        re(other.re),
        normalizedDist(other.normalizedDist),
        positiveNormalizedDist(other.positiveNormalizedDist),
        inputLayer(other.inputLayer),
        hiddenLayers(other.hiddenLayers),
        outputLayer(other.outputLayer)
    {
        reconstructLayerReferences();
    }
    DynamicNetwork& operator=(const DynamicNetwork& other)
    {
        re = other.re;
        normalizedDist = other.normalizedDist;
        positiveNormalizedDist = other.positiveNormalizedDist;
        inputLayer = other.inputLayer;
        hiddenLayers = other.hiddenLayers;
        outputLayer = other.outputLayer;
        reconstructLayerReferences();
        return *this;
    }

    DynamicNetwork& operator=(const DynamicNetwork&& other)
    {
        re = other.re;
        normalizedDist = other.normalizedDist;
        positiveNormalizedDist = other.positiveNormalizedDist;
        inputLayer = other.inputLayer;
        hiddenLayers = other.hiddenLayers;
        outputLayer = other.outputLayer;
        reconstructLayerReferences();
        return *this;
    }


    void init()
    {
        randomizeInitial(re,
                         normalizedDist,
                         normalizedDist);
    }

    void connectNetwork()
    {
        if(hiddenLayers.size() > 0)
        {
            hiddenLayers[0].connect(inputLayer, true);
            for(size_t layerIdx = 1; layerIdx < hiddenLayers.size(); ++layerIdx)
            {
                DynamicLayer& prevLayer = hiddenLayers[layerIdx-1];
                DynamicLayer& currentLayer = hiddenLayers[layerIdx];
                currentLayer.connect(prevLayer, true);
            }
            outputLayer.connect(hiddenLayers.back(), true);
        }
        init();
    }

    void process()
    {
        for(DynamicLayer& hiddenLayer : hiddenLayers)
        {
            hiddenLayer.update();
        }
        outputLayer.update();
    }

    void randomizeInitial(std::mt19937& randE,
                          std::uniform_real_distribution<float>& biasDist,
                          std::uniform_real_distribution<float>& weightDist)
    {
        for (DynamicLayer& layer : hiddenLayers)
        {
            layer.randomizeInitial(randE,
                                   biasDist,
                                   weightDist);
        }
        outputLayer.randomizeInitial(randE,
                                     biasDist,
                                     weightDist);
    }

    void mutate(float biasMutChance,
                std::normal_distribution<float>& biasMutRate,
                float weightMutChance,
                std::normal_distribution<float>& weightMutRate)
    {
        for (DynamicLayer& layer : hiddenLayers)
        {
           layer.mutate(re,
                        positiveNormalizedDist,
                        biasMutChance,
                        biasMutRate,
                        weightMutChance,
                        weightMutRate);
        }

        outputLayer.mutate(re,
                           positiveNormalizedDist,
                           biasMutChance,
                           biasMutRate,
                           weightMutChance,
                           weightMutRate);
    }

    void addHiddenLayer(size_t nbNeurons)
    {
        DynamicLayer& newHiddenLayer = hiddenLayers.emplace_back();
        newHiddenLayer.setNeuronNb(nbNeurons);
    }

    template<typename ...Ints>
    void setHiddenLayers(Ints... ints)
    {
        size_t neuronsNb[] = {ints...};
        for(size_t neuronsForLayer : neuronsNb)
        {
            addHiddenLayer(neuronsForLayer);
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

    size_t getInputNb()
    {
        return inputLayer.getNeuronNb();
    }

    void setOutputNb(size_t outputNb)
    {
        outputLayer.setNeuronNb(outputNb);
    }

    size_t getOutputNb()
    {
        return outputLayer.getNeuronNb();
    }

    void setInputValue(size_t inputId, float value)
    {
        inputLayer.setNeuronValue(inputId, value);
    }

    float getOutputValue(size_t outputId) const
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
    void reconstructLayerReferences()
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

    std::mt19937& re;
    std::uniform_real_distribution<float> normalizedDist{-1,1};
    std::uniform_real_distribution<float> positiveNormalizedDist{0,1};
    DynamicLayer inputLayer;
    std::vector<DynamicLayer> hiddenLayers;
    DynamicLayer outputLayer;
};
