#include <iostream>
#include <cmath>
#include "normalizedvalue.h"

template<typename T>
void testSpecific(float min, float max, float nMin, float nMax, float test1, float test2, float test3, float test4)
{
    std::cout << "Creating NormalizedValue<" << std::string(typeid(T).name()) << "> nv(" << min << "," << max << "," <<
                 nMin << "," << nMax << ")" << std::endl;
    NormalizedValue<T> nv(min, max, nMin, nMax);
    std::cout << "Setting to " << test1 << std::endl;
    nv.set(test1);
    std::cout << "Output of getNormalized(): " << std::fixed << nv.getNormalized() << std::endl;

    std::cout << "Setting to " << test2 << std::endl;
    nv.set(test2);
    std::cout << "Output of getNormalized(): " << std::fixed << nv.getNormalized() << std::endl;

    std::cout << "Setting to " << test3 << std::endl;
    nv.set(test3);
    std::cout << "Output of getNormalized(): " << std::fixed << nv.getNormalized() << std::endl;

    std::cout << "Setting to " << test4 << std::endl;
    nv.set(test4);
    std::cout << "Output of getNormalized(): " << std::fixed << nv.getNormalized() << std::endl;
}

template<typename T>
void testNormalization()
{
    NormalizedValue<float> nv(-1,1,-1,1);
    float initial = 0.552142521f;
    nv.setNormalized(initial);
    float result = nv.get();
    float diff = fabs(result - initial);
    if(diff > std::numeric_limits<float>::epsilon())
    {
        std::cout << "Error while converting normalized value! " << std::endl;
    }

    testSpecific<T>(0, 1000, 0, 1, 600, -600, 1600, -1600);
    testSpecific<T>(-1000, 1000, 0, 1, 600, -600, 1600, -1600);
    testSpecific<T>(50, 100, -2, 2, 50, 75, -75, 99);
    testSpecific<T>(0, 1000, -1, 1, 10, 100, 500, 501);
}


int main ([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
    testNormalization<float>();
    testNormalization<double>();
    testNormalization<long double>();
    return 0;
}
