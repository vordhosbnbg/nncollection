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
    std::minstd_rand re{rd()};

    std::normal_distribution<float> small_change(0,0.3f);
    std::vector<Net> vecNets;
    vecNets.reserve(netNb);
    for(size_t ind = 0; ind < netNb; ++ind)
    {
        vecNets.emplace_back(re);
    }
    for(Net net : vecNets)
    {
        net.template setInput<0>(0.1f);
        net.template setInput<1>(0.2f);
        net.template setInput<2>(0.3f);
        net.template setInput<3>(0.4f);
        net.template setInput<4>(0.5f);
        net.template setInput<5>(0.1f);
        net.template setInput<6>(0.2f);
        net.template setInput<7>(0.3f);
        net.template setInput<8>(0.4f);
        net.template setInput<9>(0.5f);
        net.template setInput<10>(0.1f);
        net.template setInput<11>(0.2f);
        net.template setInput<12>(0.3f);
        net.template setInput<13>(0.4f);
        net.template setInput<14>(0.5f);
        net.template setInput<15>(0.1f);
        net.template setInput<16>(0.2f);
        net.template setInput<17>(0.3f);
        net.template setInput<18>(0.4f);
        net.template setInput<19>(0.5f);
        net.template setInput<20>(0.1f);
        net.template setInput<21>(0.2f);
        net.template setInput<22>(0.3f);
        net.template setInput<23>(0.4f);
        net.template setInput<24>(0.5f);
        net.template setInput<25>(0.1f);
        net.template setInput<26>(0.2f);
        net.template setInput<27>(0.3f);
        net.template setInput<28>(0.4f);
        net.template setInput<29>(0.5f);
        net.template setInput<30>(0.1f);
        net.template setInput<30>(0.2f);
        net.template setInput<31>(0.3f);
        net.template setInput<32>(0.4f);
        net.template setInput<33>(0.5f);
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
    using DynamicNet = DynamicNetworkAdapter<78,78*2,78*3,78*2,78>;

    //doTest<StaticNet>();
    doTest<DynamicNet>();


    return 0;
}
