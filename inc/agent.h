#pragma once
#include <random>

template<typename Network>
struct Agent
{
    Agent(std::mt19937& randE) : net(randE) {}
    Network net;
    double fitness{};
};
