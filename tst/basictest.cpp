#include <iostream>
#include <random>
#include "ffnetwork.h"

int main ([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
    std::random_device rd;
    std::mt19937 re{rd()};

    FFNetwork<5,1,4,3,2> net(re);
    std::cout << "Setting inputs to 0.1, 0.2, 0.3, 0.4, 0.5" << std::endl;
    net.setInput<0>(0.1);
    net.setInput<1>(0.2);
    net.setInput<2>(0.3);
    net.setInput<3>(0.4);
    net.setInput<4>(0.5);
    net.process();
    std::cout << "Output is: " << std::fixed << net.getOutput<0>() << std::endl;
    std::cout << "Mutating (small chance)..." << std::endl;
    std::normal_distribution<float> small_change(0,0.1);
    net.mutate(0.1,
               small_change,
               0.1,
               small_change);
    net.process();
    std::cout << "Output is: " << std::fixed << net.getOutput<0>() << std::endl;
    std::cout << "Mutating (big chance)..." << std::endl;

    std::normal_distribution<float> big_change(0,0.3);
    net.mutate(0.5,
               big_change,
               0.5,
               big_change);
    net.process();
    std::cout << "Output is: " << std::fixed << net.getOutput<0>() << std::endl;

    std::uniform_real_distribution<float> biasMutationChance(0, 0.1);
    std::uniform_real_distribution<float> biasMutationRate(0, 0.1);
    std::uniform_real_distribution<float> weightMutationChance(0, 0.1);
    std::uniform_real_distribution<float> weightMutationRate(0, 0.1);

    return 0;
}
