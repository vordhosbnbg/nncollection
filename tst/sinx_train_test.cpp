#include <cmath>
#include <map>
#include <fstream>
#include <iomanip>
#include "geneticsimulation.h"
#include "ffnetwork.h"

int main ([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
    std::random_device rd;
    std::mt19937 re{rd()};
    std::uniform_real_distribution<float> simpleDist(-3.14,3.14);
    using NetTopology = FFNetwork<1 /*inputs*/,1 /*outputs*/,2 /*HL#1 neurons*/,3 /*HL#2 neurons*/,4 /*HL#3 neurons*/,5 /*HL#3 neurons*/,6 /*HL#3 neurons*/>;
    GeneticSimulation<
            NetTopology,
            2000 /*agents*/,
            200 /*keep best*/,
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

    gs.setInputRange<0>(-3.14,3.14);
    gs.setOutputRange<0>(-1,1);

    gs.train(nbEpochs,true);

    NetTopology bestNet = gs.getBestNetwork();
    float x = -3.14;
    float yExpected = 0;
    float yNet = 0;
    NormalizedValue<float> inputVal(-3.14, 3.14, -1, 1);
    std::map<float, std::pair<float,float>> plotData;
    while(x < 3.14)
    {
        yExpected = std::sin(x);

        inputVal.set(x);
        bestNet.setInput<0>(inputVal.getNormalized());
        bestNet.process();
        yNet = bestNet.getOutput<0>();
        plotData[x] = std::make_pair(yExpected, yNet);
        x += 0.01;
    }
    std::cout << "Training finished - outputing test data along with expected and actual results of best agent" << std::endl;

    std::ofstream ofs("sin_x_train_test_best_results.csv");
    for(auto it = plotData.begin(); it != plotData.end(); ++it)
    {
        float xVal = it->first;
        std::pair<float, float>& plotEntry = it->second;
        ofs << std::setprecision(2) << std::fixed << xVal << "," << plotEntry.first << "," << plotEntry.second << std::endl;
    }
    return 0;
}
