#pragma once
#include <random>

template<typename Network>
struct Agent
{
    Agent(std::minstd_rand& randE) : net(randE) {}
    Network net;
    double fitness{};
};
