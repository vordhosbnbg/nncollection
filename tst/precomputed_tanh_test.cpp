#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "precomputedtanh.h"


struct TanhData
{
    float x;
    float y;
};

int main ([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
    constexpr auto prec = PrecomputedTanh<10000,-10,10>();


    std::vector<TanhData> precomputedData;
    std::vector<TanhData> runtimeData;

    constexpr size_t numberOfElements = 2000000;

    precomputedData.resize(numberOfElements);
    runtimeData.resize(numberOfElements);

    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;

    std::cout << "Calculating tanh(x) by precomputed table " << numberOfElements << " times" << std::endl;
    t1 = std::chrono::high_resolution_clock::now();

    size_t idx = 0;
    for(double x = -10.0; x < 10.0; x += 0.00001)
    {
        precomputedData[idx].x = x;
        precomputedData[idx].y = prec.tanh(x);
        ++idx;
    }

    t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> actionTime = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
    std::cout << "Time taken for precomputed initialization: " << actionTime.count() << " s" << std::endl;
    std::cout << "Single tanh(x): " << actionTime.count() / numberOfElements * 1000000000 << " ns" << std::endl;

    std::cout << "Calculating tanh(x) by std::tanh() " << numberOfElements << " times" << std::endl;
    t1 = std::chrono::high_resolution_clock::now();

    idx = 0;
    for(double x = -10.0; x < 10.0; x += 0.00001)
    {
        runtimeData[idx].x = x;
        runtimeData[idx].y = std::tanh(x);
        ++idx;
    }

    t2 = std::chrono::high_resolution_clock::now();
    actionTime = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
    std::cout << "Time taken for runtime initialization: " << actionTime.count() << " s" << std::endl;
    std::cout << "Single tanh(x): " << actionTime.count() / numberOfElements * 1000000000 << " ns" << std::endl;

    double maxError = 0.0;
    idx=0;
    for(double x = -10.0; x < 10.0; x += 0.00001)
    {
        maxError += std::fabs(runtimeData[idx].y - precomputedData[idx].y);
        ++idx;
    }
    std::cout << "Total error " << std::fixed << maxError << std::endl;


    std::ofstream ofs("precomputed_tanh_test.csv");
    for(idx = 0; idx < precomputedData.size(); ++idx)
    {
        ofs << std::setprecision(10) << std::fixed
            << runtimeData[idx].x << ","
            << runtimeData[idx].y << ","
            << precomputedData[idx].x << ","
            << precomputedData[idx].y << "\n";
    }
    return 0;
}
