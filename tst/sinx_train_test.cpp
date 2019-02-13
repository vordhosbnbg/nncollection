#include <cmath>
#include "geneticsimulation.h"
#include "ffnetwork.h"

int main ([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
    std::random_device rd;
    std::mt19937 re{rd()};
    std::uniform_real_distribution<float> simpleDist(-1,1);
    GeneticSimulation<
            FFNetwork<1 /*inputs*/,1 /*outputs*/,10 /*HL#1 neurons*/,10 /*HL#2 neurons*/,10 /*HL#3 neurons*/>,
            100 /*agents*/,
            20 /*keep best*/,
            10 /*survival chance of rest*/> gs;
    constexpr size_t nbEntries = 1000;
    constexpr unsigned int nbEpochs= 1000;

    // prepare test data - sinf() function
    for(size_t ind = 0; ind < nbEntries; ++ind)
    {
        auto& entry = gs.testData.addEntry();
        entry.inputs[0] = simpleDist(re);
        entry.outputs[0] = std::sin(entry.inputs[0]);
    }

    gs.setInputRange<0>(-1,1);
    gs.setOutputRange<0>(-1,1);

    gs.train(nbEpochs,true);
    return 0;
}
