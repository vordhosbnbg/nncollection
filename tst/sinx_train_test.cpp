#include <cmath>
#include <map>
#include <fstream>
#include <iomanip>
#include "geneticsimulation.h"
#include "ffnetwork.h"
#include "jsonarchive.h"

int main ([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
    std::random_device rd;
    std::mt19937 re{rd()};
    std::uniform_real_distribution<float> simpleDist(-3.14,3.14);
    using NetTopology = FFNetwork<1 /*inputs*/,1 /*outputs*/,32 /*HL#1 neurons*/,32 /*HL#1 neurons*/,32 /*HL#1 neurons*/>;
    GeneticSimulation<
            NetTopology,
            1000 /*agents*/,
            100 /*keep best*/,
            5 /*survival chance of rest*/> gs;
    //constexpr size_t nbEntries = 1000;
    constexpr unsigned int nbEpochs= 5000;
    constexpr double range = 3.14127;
    // prepare test data - sinf() function
    float inpX = -range;
    while(inpX <= range)
    {
        auto& entry = gs.testData.addEntry();
        entry.inputs[0] = inpX;
        entry.outputs[0] = std::sin(entry.inputs[0]);
        inpX += 0.01;
    }

    gs.setInputRange<0>(-range,range);
    gs.setOutputRange<0>(-1,1);

    gs.train(nbEpochs,true);

    NetTopology bestNet = gs.getBestNetwork();
    float x = -range;
    float yExpected = 0;
    float yNet = 0;
    NormalizedValue<float> inputVal(-range, range, -1, 1);
    std::map<float, std::pair<float,float>> plotData;
    while(x <= range)
    {
        yExpected = std::sin(x);

        inputVal.set(x);
        bestNet.setInput<0>(inputVal.getNormalized());
        bestNet.process();
        yNet = bestNet.getOutput<0>();
        plotData[x] = std::make_pair(yExpected, yNet);
        x += 0.001;
    }
    std::cout << "Training finished - outputing test data along with expected and actual results of best agent" << std::endl;

    JSONArchive jsonArchive("bestNet.json");
    jsonArchive.write(bestNet);
    NetTopology net2 = gs.getNetworkByNumber(1);
    NetTopology net5 = gs.getNetworkByNumber(4);
    NetTopology net10 = gs.getNetworkByNumber(9);

    JSONArchive jsonNet2("net02.json");
    jsonNet2.write(net2);
    JSONArchive jsonNet5("net05.json");
    jsonNet5.write(net5);
    JSONArchive jsonNet10("net10.json");
    jsonNet10.write(net10);


    std::ofstream ofs("sin_x_train_test_best_results.csv");
    for(auto it = plotData.begin(); it != plotData.end(); ++it)
    {
        float xVal = it->first;
        std::pair<float, float>& plotEntry = it->second;
        ofs << std::setprecision(10) << std::fixed << xVal << "," << plotEntry.first << "," << plotEntry.second << std::endl;
    }
    gs.exportStatisticsToCSV("sin_x_train_test_epoch_stats.csv");
    return 0;
}
