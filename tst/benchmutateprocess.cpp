#include <iostream>
#include <chrono>
#include "ffnetwork.h"


int main ([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
    constexpr size_t netNb = 10000;
    constexpr size_t processIterations = 1000;
    constexpr size_t mutateIterations = 1000;
    std::random_device rd;
    std::mt19937 re{rd()};

    using Net = FFNetwork<5,1,4,3,2>;

    std::uniform_real_distribution<float> small_change(-0.1,0.1);
    std::vector<Net> vecNets;
    vecNets.reserve(netNb);
    for(size_t ind = 0; ind < netNb; ++ind)
    {
        vecNets.emplace_back(Net(re));
    }
    for(Net net : vecNets)
    {
        net.setInput<0>(0.1);
        net.setInput<1>(0.2);
        net.setInput<2>(0.3);
        net.setInput<3>(0.4);
        net.setInput<4>(0.5);
    }

    std::chrono::high_resolution_clock::time_point previous = std::chrono::high_resolution_clock::now();
    for(Net net : vecNets)
    {
        for(size_t ind = 0; ind < mutateIterations; ++ind)
        {
            net.mutate(0.1,
                       small_change,
                       0.1,
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


    return 0;
}
