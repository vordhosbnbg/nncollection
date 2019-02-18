#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
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
        hwConcurency = std::thread::hardware_concurrency();
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

    void trainAgentRange(unsigned int threadId, size_t agentMin, size_t agentMax)
    {
        while(!wkThreadsCanExit.load())
        {
            std::unique_lock<std::mutex> lk(threadStartMutex);
            cvSync.wait(lk, [&]{ return wkThreadsCanStart[threadId] == true; });

            wkThreadsCanStart[threadId] = false;
            if(wkThreadsCanExit.load())
                return;
            lk.unlock();
            for(const typename decltype(testData.data)::value_type& entry : testData.data)
            {
                setAllAgentInputs(entry, agentMin, agentMax);
                processAllAgents(agentMin, agentMax);
                evaluateFitnessForAllAgents(entry, agentMin, agentMax);
            }
            {
                std::lock_guard<std::mutex> lk2(threadEndMutex);
                nbThreadsWorking--;
                //std::cerr << "Thread finished - remaining workers " << nbThreadsWorking << std::endl;
            }
            cvSync.notify_all();
        }
    }


    void train(unsigned int epochs, bool printDebugInfo = false)
    {
        printInfo = printDebugInfo;
        wkThreadsCanStart.resize(hwConcurency+1, false);
        wkThreadsCanExit.store(false);

        for(unsigned int threadId = 0; threadId <= hwConcurency; ++threadId)
        {
            //std::cout << "Starting thread with id " << threadId << std::endl;
            size_t agentRange = agents.size() / hwConcurency;
            size_t agentMin = agentRange * threadId;
            if(threadId == hwConcurency)
            {
                agentRange = agents.size() % hwConcurency;
            }
            size_t agentMax = agentMin + agentRange;
            //std::cout << "Processing agents from " << agentMin << " to " << agentMax-1 << std::endl;

            workerThreads.emplace_back([&, threadId, agentMin, agentMax]{trainAgentRange(threadId, agentMin, agentMax);});
        }

        for(unsigned int currentEpoch = 0; currentEpoch <= epochs ; ++currentEpoch)
        {
            setMaxFitnessForAgents(testData.data.size());
            std::chrono::high_resolution_clock::time_point t1;
            std::chrono::high_resolution_clock::time_point t2;

            if(printInfo)
            {
                t1 = std::chrono::high_resolution_clock::now();
            }

            {
                std::lock_guard<std::mutex> lk(threadStartMutex);
                std::lock_guard<std::mutex> lk2(threadEndMutex);
                nbThreadsWorking = hwConcurency + 1;
                for(size_t idx = 0; idx < wkThreadsCanStart.size(); ++idx)
                {
                    wkThreadsCanStart[idx] = true;
                }
                if(currentEpoch == epochs)
                {
                    wkThreadsCanExit.store(true);
                    cvSync.notify_all();
                    for(std::thread& workerThread : workerThreads)
                    {
                        workerThread.detach();
                    }
                    return;
                }
                //std::cerr << "Threads can start working " << std::endl;
            }
            cvSync.notify_all();

            std::unique_lock<std::mutex> lk(threadEndMutex);
            cvSync.wait(lk, [&]{ return nbThreadsWorking == 0; });

            sortAllAgentsOnFitness();
            saveStats();
            if(printInfo)
            {
                t2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> actionTime = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
                printEpochStatistics(currentEpoch);
                printTimeForProcessingEpoch(actionTime.count());
            }
            removeWorstAndCloneBestWithMutation();
        }

    }

    Network getNetworkByNumber(size_t nb)
    {
        if(nb >= agents.size())
        {
            nb = agents.size() - 1;
        }
        return agents[nb].net;
    }

    Network getBestNetwork()
    {
        return agents[0].net;
    }

    void exportStatisticsToCSV(const std::string& filename)
    {
        std::ofstream ofs(filename);
        for(const EpochStatistics& stats : epochStatistics)
        {
            ofs << std::setprecision(10) << std::fixed <<
                   stats.maxFitnessDataset << "," <<
                   stats.singleBestDatasetFintess << "," <<
                   stats.avgFitnessDatasetBest << "," <<
                   stats.avgFitnessDatasetRest << "," <<
                   stats.maxFintessEntry << "," <<
                   stats.singleBestEntryFitness << "," <<
                   stats.avgFitnessEntryBest << "," <<
                   stats.avgFitnessEntryRest << "," <<
                   std::endl;
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

    void evaluateFitnessForAllAgents(const typename decltype(testData.data)::value_type& entry, size_t agentMin, size_t agentMax)
    {
        for(size_t idx = agentMin; idx < agentMax; ++idx)
        {
            Agent<Network>& agent = agents[idx];
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

    void setAllAgentInputs(const typename decltype(testData.data)::value_type& entry, size_t agentMin, size_t agentMax)
    {
        for(size_t idx = agentMin; idx < agentMax; ++idx)
        {
            Agent<Network>& agent = agents[idx];
            setInputsForAgent(agent, entry);
        }
    }

    void processAllAgents(size_t agentMin, size_t agentMax)
    {
        for(size_t idx = agentMin; idx < agentMax; ++idx)
        {
            Agent<Network>& agent = agents[idx];
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
        std::normal_distribution<float> mutRate(0,0.25);
        std::normal_distribution<float> mutRateStable(0,0.15);
        std::uniform_int_distribution<unsigned int> worstSurvivalChance(0,100);
        nbRestSurvived = 0;

        for(size_t index = keepBestNb; index < agentsNb; ++index)
        {
            if(bestIndex>keepBestNb)
            {
                bestIndex = 0;
            }
            unsigned int canPoorAgentSurvive = 100;
            if(restSurvivalChance > 0)
            {
                canPoorAgentSurvive = worstSurvivalChance(re);
            }
            if(canPoorAgentSurvive >= restSurvivalChance)
            {
                // tough chance - we obliterate it and replace it with a mutated copy of one of the best
                agents[index] = agents[bestIndex];
                agents[index].net.mutate(0.05, mutRate, 0.05, mutRate);
                ++bestIndex;
            }
            else
            {
                // it survived but has to mutate
                agents[index].net.mutate(0.2, mutRate, 0.2, mutRate);
                nbRestSurvived++;
            }
        }
    }

    void printTimeForProcessingEpoch(double timeInSec)
    {
        std::cout << "Total time for epoch: " << std::fixed << timeInSec << " s" << std::endl;
        std::cout << "Time per agent per data entry: " << std::fixed << (timeInSec / agentsNb * 1000000000 / testData.data.size()) << " ns" << std::endl;
    }

    void saveStats()
    {
        EpochStatistics stats;
        stats.maxFitnessDataset = maxFitness;
        stats.singleBestDatasetFintess = agents[0].fitness;
        stats.avgFitnessDatasetBest = getAverageFitnessFromBest();
        stats.avgFitnessDatasetRest = getAverageFitnessFromRest();
        stats.maxFintessEntry = maxFitness / testData.data.size();
        stats.singleBestEntryFitness = stats.singleBestDatasetFintess / testData.data.size();
        stats.avgFitnessEntryBest = stats.avgFitnessDatasetBest / testData.data.size();
        stats.avgFitnessEntryRest = stats.avgFitnessDatasetRest / testData.data.size();
        epochStatistics.emplace_back(stats);
    }

    void printFitnessInfo()
    {
        EpochStatistics& stats = epochStatistics.back();
        std::cout << "MaxFitness (dataset): " <<  std::setprecision(10) <<  std::fixed << stats.maxFitnessDataset << std::endl;
        std::cout << "Average fitness from best (dataset): " << keepBestNb << " agents: " << std::fixed << stats.avgFitnessDatasetBest << std::endl;
        std::cout << "Average fitness from rest (dataset): " << (agentsNb - keepBestNb) << " agents: " << std::fixed << stats.avgFitnessDatasetRest << std::endl;
        std::cout << "MaxFitness (entry): " << std::fixed << stats.maxFintessEntry << std::endl;
        std::cout << "Average fitness from best (entry): " << keepBestNb << " agents: " << std::fixed << stats.avgFitnessEntryBest << std::endl;
        std::cout << "Average fitness from rest (entry): " << (agentsNb - keepBestNb) << " agents: " << std::fixed << stats.avgFitnessEntryRest << std::endl;
        std::cout << "\nSinge best fitness (dataset): " << std::fixed << stats.singleBestDatasetFintess << std::endl;
        std::cout << "Singe best fitness (entry): " << std::fixed << stats.singleBestEntryFitness << std::endl;
    }

    void printEpochStatistics(unsigned int epochN)
    {
        std::cout << "\n\n================ Epoch " << epochN << "================" << std::endl;
        std::cout << nbRestSurvived << " lucky agents survived despite doing poorly" << std::endl;
        printFitnessInfo();
    }

    struct EpochStatistics
    {
        float maxFitnessDataset;
        float singleBestDatasetFintess;
        float avgFitnessDatasetBest;
        float avgFitnessDatasetRest;
        float maxFintessEntry;
        float singleBestEntryFitness;
        float avgFitnessEntryBest;
        float avgFitnessEntryRest;
    };

    std::random_device rd;
    std::mt19937 re{rd()};
    std::vector<Agent<Network>> agents;
    std::vector<NormalizedValue<float>> inputs;
    std::vector<NormalizedValue<float>> outputs;
    unsigned int hwConcurency;
    std::vector<std::thread> workerThreads;
    bool printInfo{false};
    unsigned int nbRestSurvived{};
    float maxFitness{};
    std::vector<EpochStatistics> epochStatistics;
    std::condition_variable cvSync;
    std::mutex threadStartMutex;
    std::mutex threadEndMutex;
    std::vector<bool> wkThreadsCanStart;
    unsigned int nbThreadsWorking{0};
    std::atomic<bool> wkThreadsCanExit;
};


