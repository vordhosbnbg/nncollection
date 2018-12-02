#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include "agent.h"
#include "normalizedvalue.h"
#include "testdata.h"

template<typename Network,
         size_t agentsNb,
         size_t keepBestNb,
         unsigned int restSurvivalChance>
class GeneticSimulation
{
public:
    GeneticSimulation()
    {
        agents.reserve(agentsNb);
        for(size_t ind = 0; ind < agentsNb; ++ind)
        {
            agents.emplace_back(Agent<Network>(re));
        }
        inputs.reserve(Network::getInputNb());
        for(unsigned int ind = 0; ind < Network::getInputNb(); ++ind)
        {
            inputs.emplace_back(NormalizedValue<float>(-1,1,-1,1));
        }
        outputs.reserve(Network::getOutputNb());
        for(unsigned int ind = 0; ind < Network::getOutputNb(); ++ind)
        {
            outputs.emplace_back(NormalizedValue<float>(-1,1,-1,1));
        }
    }

    template<unsigned int inputId>
    void setInputRange(float min, float max)
    {
        inputs[inputId].setMin(min);
        inputs[inputId].setMax(max);
    }

    template<unsigned int outputId>
    void setOutputRange(float min, float max)
    {
        outputs[outputId].setMin(min);
        outputs[outputId].setMax(max);
    }


    void train(unsigned int epochs, bool printDebugInfo = false)
    {
        printInfo = printDebugInfo;
        for(unsigned int currentEpoch = 0; currentEpoch < epochs ; ++currentEpoch)
        {
            setMaxFitnessForAgents(testData.data.size());
            std::chrono::high_resolution_clock::time_point t1;
            std::chrono::high_resolution_clock::time_point t2;
            if(printInfo)
            {
                t1 = std::chrono::high_resolution_clock::now();
            }
            for(const typename decltype(testData.data)::value_type& entry : testData.data)
            {
                setAllAgentInputs(entry);
                processAllAgents();
                evaluateFitnessForAllAgents(entry);
            }

            if(printInfo)
            {
                t2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> actionTime = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
                printEpochStatistics(currentEpoch);
                printTimeForProcessingEpoch(actionTime.count());
            }
            sortAllAgentsOnFitness();
            removeWorstAndCloneBestWithMutation();
        }
    }

    TestData<float, Network::getInputNb(), Network::getOutputNb()> testData;

private:

    float getMaxFitness()
    {
        float fitness = 0;
        for(const NormalizedValue<float>& output : outputs)
        {
            fitness += output.getMax() - output.getMin();
        }
        return fitness;
    }

    float maxFitness{};

    template<unsigned int outputId>
    typename std::enable_if<outputId == Network::getOutputNb()>::type
    setAndGetOutputAndRecurse([[maybe_unused]] const Agent<Network>& agent,
                              [[maybe_unused]] std::array<float,Network::getOutputNb()>& actualOutputs)
    {}

    template<unsigned int outputId>
    typename std::enable_if<outputId < Network::getOutputNb()>::type
    setAndGetOutputAndRecurse(const Agent<Network>& agent, std::array<float,Network::getOutputNb()>& actualOutputs)
    {
        NormalizedValue<float>& output = outputs[outputId];
        output.setNormalized(agent.net.template getOutput<outputId>());
        actualOutputs[outputId] = output.get();
        setAndGetOutputAndRecurse<outputId+1>(agent, actualOutputs);
    }

    void evaluateFitness(Agent<Network>& agent, const typename decltype(testData.data)::value_type& entry)
    {
        std::array<float,Network::getOutputNb()> actualOutputs;
        setAndGetOutputAndRecurse<0>(agent,actualOutputs);
        for(unsigned int ind = 0; ind < actualOutputs.size(); ++ind)
        {
            agent.fitness -= std::fabs(entry.outputs[ind] - actualOutputs[ind]);
        }
    }

    void evaluateFitnessForAllAgents(const typename decltype(testData.data)::value_type& entry)
    {
        for(Agent<Network>& agent : agents)
        {
            evaluateFitness(agent, entry);
        }
    }

    template<unsigned int inputId>
    typename std::enable_if<inputId == Network::getInputNb()>::type
    setInputAndRecurse([[maybe_unused]] Agent<Network>& agent,
                       [[maybe_unused]] const std::array<float, Network::getInputNb()>& entryInputs)
    {
    }

    template<unsigned int inputId>
    typename std::enable_if<inputId < Network::getInputNb()>::type
    setInputAndRecurse(Agent<Network>& agent, const std::array<float, Network::getInputNb()>& entryInputs)
    {
        inputs[inputId].set(entryInputs[inputId]);
        agent.net.template setInput<inputId>(inputs[inputId].getNormalized());
        setInputAndRecurse<inputId+1>(agent, entryInputs);
    }
    void setInputsForAgent(Agent<Network>& agent, const typename decltype(testData.data)::value_type& entry)
    {
        const std::array<float, Network::getInputNb()>& entryInputs = entry.inputs;
        setInputAndRecurse<0>(agent, entryInputs);
    }

    void setAllAgentInputs(const typename decltype(testData.data)::value_type& entry)
    {
        for(Agent<Network>& agent : agents)
        {
            setInputsForAgent(agent, entry);
        }
    }

    void processAllAgents()
    {
        for(Agent<Network>& agent : agents)
        {
            agent.net.process();
        }
    }

    void setMaxFitnessForAgents(unsigned int dataSetSize)
    {
        maxFitness = getMaxFitness();
        maxFitness *= dataSetSize;
        for(Agent<Network>& agent : agents)
        {
            agent.fitness = maxFitness;
        }
    }

    void sortAllAgentsOnFitness()
    {
        std::sort(agents.begin(), agents.end(),
                  [](const Agent<Network>& a, const Agent<Network>& b) -> bool
        {
            return a.fitness > b.fitness;
        });
    }

    float getAverageFitnessFromBest()
    {
        float avgFitness{};
        for(size_t index = 0; index < keepBestNb; ++index)
        {
            avgFitness+= agents[index].fitness;
        }
        avgFitness /= keepBestNb;
        return avgFitness;
    }

    float getAverageFitnessFromRest()
    {
        float avgFitness{};
        for(size_t index = keepBestNb; index < agentsNb; ++index)
        {
            avgFitness+= agents[index].fitness;
        }
        avgFitness /= (agentsNb - keepBestNb);
        return avgFitness;
    }

    void removeWorstAndCloneBestWithMutation()
    {
        size_t bestIndex = 0;
        std::uniform_real_distribution<float> mutRate(-1,1);
        std::uniform_int_distribution<unsigned int> worstSurvivalChance(0,100);
        nbRestSurvived = 0;

        for(size_t index = keepBestNb; index < agentsNb; ++index)
        {
            if(bestIndex>keepBestNb)
            {
                bestIndex = 0;
            }
            unsigned int canPoorAgentSurvive = worstSurvivalChance(re);
            if(canPoorAgentSurvive > restSurvivalChance)
            {
                // tough chance - we obliterate it and replace it with a mutated copy of one of the best
                agents[index] = agents[bestIndex];
                agents[index].net.mutate(0.1, mutRate, 0.1, mutRate);
                ++bestIndex;
            }
            else
            {
                // it survived but has to mutate
                agents[index].net.mutate(0.1, mutRate, 0.1, mutRate);
                nbRestSurvived++;
            }
        }
    }

    void printTimeForProcessingEpoch(double timeInSec)
    {
        std::cout << "Total time for epoch: " << std::fixed << timeInSec << " s" << std::endl;
        std::cout << "Time per agent per data entry: " << std::fixed << (timeInSec / agentsNb * 1000000000 / testData.data.size()) << " ns" << std::endl;
    }

    void printFitnessInfo()
    {
        std::cout << "MaxFitness (dataset): " << std::fixed << maxFitness << std::endl;
        std::cout << "Average fitness from best (dataset): " << keepBestNb << " agents: " << std::fixed << getAverageFitnessFromBest() << std::endl;
        std::cout << "Average fitness from rest (dataset): " << (agentsNb - keepBestNb) << " agents: " << std::fixed << getAverageFitnessFromRest() << std::endl;
        std::cout << "Average MaxFitness (entry): " << std::fixed << (maxFitness / testData.data.size()) << std::endl;
        std::cout << "Average fitness from best (entry): " << keepBestNb << " agents: " << std::fixed << (getAverageFitnessFromBest() / testData.data.size()) << std::endl;
        std::cout << "Average fitness from rest (entry): " << (agentsNb - keepBestNb) << " agents: " << std::fixed << (getAverageFitnessFromRest() / testData.data.size()) << std::endl;
        std::cout << "\nSinge best fitness (dataset): " << std::fixed << agents[0].fitness << std::endl;
        std::cout << "Singe best fitness (entry): " << std::fixed << (agents[0].fitness / testData.data.size()) << std::endl;
    }

    void printEpochStatistics(unsigned int epochN)
    {
        std::cout << "\n\n================ Epoch " << epochN << "================" << std::endl;
        std::cout << nbRestSurvived << " lucky agents survived despite doing poorly" << std::endl;
        printFitnessInfo();
    }


    std::random_device rd;
    std::mt19937 re{rd()};
    std::vector<Agent<Network>> agents;
    std::vector<NormalizedValue<float>> inputs;
    std::vector<NormalizedValue<float>> outputs;
    bool printInfo{false};
    unsigned int nbRestSurvived{};
};
