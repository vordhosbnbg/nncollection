#include <iostream>
#include <chrono>
#include "ffnetwork.h"
#include "dnetworkadapter.h"


template <typename Net>
void doTest()
{
    constexpr size_t netNb = 100;
    constexpr size_t processIterations = 1000;
    constexpr size_t mutateIterations = 10;
    std::random_device rd;
    std::mt19937 re{rd()};

    std::normal_distribution<float> small_change(0,0.3f);
    std::vector<Net> vecNets;
    vecNets.reserve(netNb);
    for(size_t ind = 0; ind < netNb; ++ind)
    {
        vecNets.emplace_back(re);
    }
    for(Net net : vecNets)
    {
        net.template setInput<0>(0.1);
        net.template setInput<1>(0.2);
        net.template setInput<2>(0.3);
        net.template setInput<3>(0.4);
        net.template setInput<4>(0.5);
    }

    std::chrono::high_resolution_clock::time_point previous = std::chrono::high_resolution_clock::now();
    for(Net net : vecNets)
    {
        for(size_t ind = 0; ind < mutateIterations; ++ind)
        {
            net.mutate(0.1f,
                       small_change,
                       0.1f,
                       small_change);
        }
    }
    std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> actionTime = std::chrono::duration_cast<std::chrono::microseconds>(now-previous);
    double totalMutateTime = actionTime.count();
    double mutateTimeSingle = 1000000000 * (totalMutateTime / netNb) / mutateIterations;
    std::cout << "Mutated "
              << netNb << " neural networks "
              << mutateIterations << " times for "
              << std::fixed << totalMutateTime << " sec." << std::endl;

    std::cout << "Single mutate time - " << std::fixed << mutateTimeSingle << " ns " << std::endl;
    previous = std::chrono::high_resolution_clock::now();
    for(Net net : vecNets)
    {
        for(size_t ind = 0; ind < processIterations; ++ind)
        {
            net.process();
        }
    }
    now = std::chrono::high_resolution_clock::now();
    actionTime = std::chrono::duration_cast<std::chrono::microseconds>(now-previous);
    double totalProcessTime = actionTime.count();
    double processTimeSingle = 1000000000 * (totalProcessTime / netNb) / processIterations;
    std::cout << "Processed "
              << netNb << " neural networks "
              << processIterations << " times for "
              << std::fixed << totalProcessTime << " sec." << std::endl;
    std::cout << "Single process time - " << std::fixed << processTimeSingle << " ns " << std::endl;

}

int main ([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{

    //using StaticNet = FFNetwork<5,10,20,10,2>;
    using DynamicNet = DynamicNetworkAdapter<5,10,20,10,2>;

    //doTest<StaticNet>();
    doTest<DynamicNet>();


    return 0;
}
