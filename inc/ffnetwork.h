#pragma once

#include <array>
#include <tuple>
#include <random>
#include "layer.h"

template <unsigned int inputNb, unsigned int outputNb, unsigned int... hiddenNb>
class FFNetwork
{
public:
    constexpr FFNetwork(std::mt19937& randomEngine) : re(randomEngine)
    {
        addInputsToHiddenLayerRecurse<0>(inputLayer);
        randomizeInitial(re,
                         _normalizedDist,
                         _normalizedDist);
    }
    ~FFNetwork() = default;
    FFNetwork(const FFNetwork& other) :
        re(other.re),
        inputLayer(other.inputLayer),
        outputLayer(other.outputLayer),
        hiddenLayers(other.hiddenLayers),
        _normalizedDist(other._normalizedDist),
        _positiveNormalizedDist(other._normalizedDist)
    {
    }
    FFNetwork& operator=(const FFNetwork& other)
    {
        re = other.re;
        inputLayer = other.inputLayer;
        outputLayer = other.outputLayer;
        hiddenLayers = other.hiddenLayers;
        _normalizedDist = other._normalizedDist;
        _positiveNormalizedDist = other._normalizedDist;
        return *this;
    }


    constexpr void process()
    {
        processHiddenLayerRecurse<0, inputNb>(inputLayer);
    }

    void mutate(float biasMutChance,
                std::uniform_real_distribution<float>& biasMutRate,
                float weightMutChance,
                std::uniform_real_distribution<float>& weightMutRate)
    {
        mutateLayerAndRecurse<0>(biasMutChance,
                                 biasMutRate,
                                 weightMutChance,
                                 weightMutRate);
        outputLayer.mutate(re,
                           _positiveNormalizedDist,
                           biasMutChance,
                           biasMutRate,
                           weightMutChance,
                           weightMutRate);
    }

    template<unsigned int outputId>
    float getOutput() const
    {
        static_assert (outputId < outputNb, "outputId must be less than outputNb");
        return outputLayer.neurons[outputId].getValue();
    }

    template<unsigned int inputId>
    float setInput(float val)
    {
        static_assert (inputId < inputNb, "inputId must be less than inputNb");
        return outputLayer.neurons[inputId].setValue(val);
    }

    static constexpr unsigned int getInputNb()
    {
        return inputNb;
    }

    static constexpr unsigned int getOutputNb()
    {
        return  outputNb;
    }

private:
    std::mt19937& re;
    Layer<inputNb> inputLayer;
    Layer<outputNb> outputLayer;
    std::tuple<Layer<hiddenNb>...> hiddenLayers;
    static constexpr size_t hiddenLayersCount = std::tuple_size<decltype(hiddenLayers)>::value;
    std::uniform_real_distribution<float> _normalizedDist{-1,1};
    std::uniform_real_distribution<float> _positiveNormalizedDist{0,1};

    template<size_t layerNb>
    typename std::enable_if<layerNb == hiddenLayersCount>::type
    mutateLayerAndRecurse([[maybe_unused]] float biasMutChance,
                          [[maybe_unused]] std::uniform_real_distribution<float>& biasMutRate,
                          [[maybe_unused]] float weightMutChance,
                          [[maybe_unused]] std::uniform_real_distribution<float>& weightMutRate)
    {
    }

    template<size_t layerNb>
    typename std::enable_if<layerNb < hiddenLayersCount>::type
    mutateLayerAndRecurse(float biasMutChance,
                                 std::uniform_real_distribution<float>& biasMutRate,
                                 float weightMutChance,
                                 std::uniform_real_distribution<float>& weightMutRate)
    {
        auto& hiddenLayerCurrent = std::get<layerNb>(hiddenLayers);
        hiddenLayerCurrent.mutate(re,
                                  _positiveNormalizedDist,
                                  biasMutChance,
                                  biasMutRate,
                                  weightMutChance,
                                  weightMutRate);

        mutateLayerAndRecurse<layerNb+1>(biasMutChance,
                                         biasMutRate,
                                         weightMutChance,
                                         weightMutRate);
    }

    template<size_t layerNb>
    typename std::enable_if<layerNb == hiddenLayersCount>::type
    randomizeLayerInitialAndRecurse([[maybe_unused]] std::mt19937& randE,
                                           [[maybe_unused]] std::uniform_real_distribution<float>& biasDist,
                                           [[maybe_unused]] std::uniform_real_distribution<float>& weightDist)
    {
    }

    template<size_t layerNb>
    typename std::enable_if<layerNb < hiddenLayersCount>::type
    randomizeLayerInitialAndRecurse(std::mt19937& randE,
                                           std::uniform_real_distribution<float>& biasDist,
                                           std::uniform_real_distribution<float>& weightDist)
    {
        auto& hiddenLayerCurrent = std::get<layerNb>(hiddenLayers);
        hiddenLayerCurrent.randomizeInitial(randE,
                                            biasDist,
                                            weightDist);
        randomizeLayerInitialAndRecurse<layerNb+1>(randE,
                                                   biasDist,
                                                   weightDist);
    }

    void randomizeInitial(std::mt19937& randE,
                                 std::uniform_real_distribution<float>& biasDist,
                                 std::uniform_real_distribution<float>& weightDist)
    {
        randomizeLayerInitialAndRecurse<0>(randE,
                                           biasDist,
                                           weightDist);
        outputLayer.randomizeInitial(randE,
                                     biasDist,
                                     weightDist);
    }

    template<size_t layerNb, size_t prevLayerNeuronNb>
    typename std::enable_if<layerNb == hiddenLayersCount>::type
    constexpr processHiddenLayerRecurse(const Layer<prevLayerNeuronNb>& prevLayer)
    {
        outputLayer.update(prevLayer);
    }

//    template<size_t layerNb, size_t prevLayerNeuronNb>
//    typename std::enable_if<(layerNb == 0) && (layerNb < hiddenLayersCount)>::type
//    constexpr processHiddenLayerRecurse(const Layer<prevLayerNeuronNb>& prevLayer)
//    {

//        auto& hiddenLayerCurrent = std::get<layerNb>(hiddenLayers);

//        hiddenLayerCurrent.update(inputLayer);
//        processHiddenLayerRecurse<layerNb+1>(inputLayer);
//    }

    template<size_t layerNb, size_t prevLayerNeuronNb>
    typename std::enable_if< layerNb < hiddenLayersCount >::type
    constexpr processHiddenLayerRecurse(const Layer<prevLayerNeuronNb>& prevLayer)
    {
        auto& hiddenLayerCurrent = std::get<layerNb>(hiddenLayers);

        hiddenLayerCurrent.update(prevLayer);
        processHiddenLayerRecurse<layerNb+1>(hiddenLayerCurrent);
    }

    template<size_t layerNb, size_t prevLayerNeuronNb>
    typename std::enable_if< layerNb == hiddenLayersCount >::type
    constexpr addInputsToHiddenLayerRecurse(const Layer<prevLayerNeuronNb>& prevLayer)
    {
        outputLayer.addInputs(prevLayer);
    }

    template<size_t layerNb, size_t prevLayerNeuronNb>
    typename std::enable_if< layerNb < hiddenLayersCount >::type
    constexpr addInputsToHiddenLayerRecurse(const Layer<prevLayerNeuronNb>& prevLayer)
    {
        auto& hiddenLayerCurrent = std::get<layerNb>(hiddenLayers);

        hiddenLayerCurrent.addInputs(prevLayer);
        addInputsToHiddenLayerRecurse<layerNb+1>(hiddenLayerCurrent);
    }


};
