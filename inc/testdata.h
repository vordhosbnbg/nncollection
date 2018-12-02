#pragma once
#include <vector>
#include <array>

template<typename DataType, unsigned int inputNb, unsigned int outputNb>
struct TestData
{
    struct DataEntry
    {
        std::array<DataType, inputNb> inputs;
        std::array<DataType, outputNb> outputs;
    };

    DataEntry& addEntry()
    {
        return data.emplace_back();
    }
    std::vector<DataEntry> data;
};
