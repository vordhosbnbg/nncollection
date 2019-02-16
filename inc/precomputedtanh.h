#pragma once
#include <cmath>
#include <array>

template<size_t fractionsPerDigit, int minRange, int maxRange>
class PrecomputedTanh
{
public:
    constexpr PrecomputedTanh()
    {
        for(size_t idx = 0; idx < totalFractions; idx +=10)
        {

            float x1 = ((float)idx / (float)fractionsPerDigit) + (float)minRange;
            float x2 = ((float)(idx+1) / (float)fractionsPerDigit) + (float)minRange;
            float x3 = ((float)(idx+2) / (float)fractionsPerDigit) + (float)minRange;
            float x4 = ((float)(idx+3) / (float)fractionsPerDigit) + (float)minRange;
            float x5 = ((float)(idx+4) / (float)fractionsPerDigit) + (float)minRange;
            float x6 = ((float)(idx+5) / (float)fractionsPerDigit) + (float)minRange;
            float x7 = ((float)(idx+6) / (float)fractionsPerDigit) + (float)minRange;
            float x8 = ((float)(idx+7) / (float)fractionsPerDigit) + (float)minRange;
            float x9 = ((float)(idx+8) / (float)fractionsPerDigit) + (float)minRange;
            float x10 = ((float)(idx+9) / (float)fractionsPerDigit) + (float)minRange;
            _data[idx] = std::tanh(x1);
            _data[idx+1] = std::tanh(x2);
            _data[idx+2] = std::tanh(x3);
            _data[idx+3] = std::tanh(x4);
            _data[idx+4] = std::tanh(x5);
            _data[idx+5] = std::tanh(x6);
            _data[idx+6] = std::tanh(x7);
            _data[idx+7] = std::tanh(x8);
            _data[idx+8] = std::tanh(x9);
            _data[idx+9] = std::tanh(x10);
        }
    }

    constexpr float tanh(float x) const
    {
        if(x < minRange)
        {
            x = minRange;
        }
        else if(x > maxRange)
        {
            x = maxRange;
        }
        size_t idx = (x - minRange) * fractionsPerDigit;
        return _data[idx];
    }
private:
    static constexpr size_t totalFractions = (maxRange - minRange) * fractionsPerDigit;
    std::array<float, totalFractions> _data{};
};
