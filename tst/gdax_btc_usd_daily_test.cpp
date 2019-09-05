#include <cmath>
#include <map>
#include <fstream>
#include <iomanip>
#include "geneticsimulation.h"
#include "ffnetwork.h"
#include "dnetworkadapter.h"
#include "jsonarchive.h"
#include "csvread.h"


int main (int argc, char** argv)
{
    if(argc >= 2 && argc <= 3)
    {
        std::string inputCsv(argv[1]);
        std::ifstream ifs(inputCsv);
        std::string label;

        if(!ifs.is_open())
        {
            std::cout << "Can't open \'" << inputCsv << "\'" << std::endl;
        }
        else
        {
            std::string netFilename;
            if(argc == 3)
            {
                netFilename = argv[2];
            }

            struct BtcData
            {
                float price;
                float volumeFrom;
                float volumeTo;
            };
            std::vector<BtcData> btcData;
            CSVRow row;
            ifs >> row; // first line
            while(ifs >> row)
            {
                btcData.emplace_back();
                BtcData& btcEntry = btcData.back();
                btcEntry.price = std::atof(row[1].c_str());
                btcEntry.volumeFrom = std::atof(row[2].c_str());
                btcEntry.volumeTo = std::atof(row[3].c_str());
            }


            std::random_device rd;
            std::minstd_rand re{rd()};
            std::uniform_real_distribution<float> simpleDist(-3.14,3.14);
            constexpr unsigned int nbEpochs = 7500;
//            using NetTopology = FFNetwork<90 /*inputs*/,
//                                          1 /*outputs*/,
//                                          30 /*HL#1 neurons*/,
//                                          30 /*HL#1 neurons*/,
//                                          30 /*HL#1 neurons*/,
//                                          30 /*HL#1 neurons*/,
//                                          30 /*HL#1 neurons*/,
//                                          30 /*HL#1 neurons*/,
//                                          30 /*HL#1 neurons*/>;
            using NetTopology = DynamicNetworkAdapter<90 /*inputs*/,
                                          1 /*outputs*/,
                                          30 /*HL#1 neurons*/,
                                          30 /*HL#1 neurons*/,
                                          30 /*HL#1 neurons*/,
                                          30 /*HL#1 neurons*/,
                                          30 /*HL#1 neurons*/,
                                          30 /*HL#1 neurons*/,
                                          30 /*HL#1 neurons*/>;
            using GenSim =  GeneticSimulation<
                            NetTopology,
                            1000 /*agents*/,
                            500 /*keep best*/,
                            50 /*survival chance of rest*/>;

            std::unique_ptr<GenSim> gsPtr;

            if(!netFilename.empty())
            {
                NetTopology inputNet{re};
                JSONArchive arch(netFilename);
                if(arch.read(inputNet))
                {
                    gsPtr = std::make_unique<GenSim>(inputNet);
                }
                else
                {
                    std::cout << "Error while loading network from: " << netFilename << std::endl;
                }
            }
            else
            {
                gsPtr = std::make_unique<GenSim>();
            }

            GenSim& gs = *gsPtr.get();
            // prepare test data - sinf() function
            for(size_t idx = 30; idx < btcData.size(); ++idx)
            {
                auto& entry = gs.testData.addEntry();
                for(int daysBack = 0; daysBack < 30; ++daysBack)
                {
                    entry.inputs[daysBack*3+0] = btcData[idx-daysBack].price;
                    entry.inputs[daysBack*3+1] = btcData[idx-daysBack].volumeFrom;
                    entry.inputs[daysBack*3+2] = btcData[idx-daysBack].volumeTo;
                }
                entry.outputs[0] = btcData[idx].price;
            }

            constexpr float minBtcRange = 0;
            constexpr float maxBtcRange = 30000;
            constexpr float minVolRange = 0;
            constexpr float maxVolRange = 1300000000;
            {
                gs.setInputRange<0+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<0+1>(minVolRange,maxVolRange);
                gs.setInputRange<0+2>(minVolRange,maxVolRange);
                gs.setInputRange<1*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<1*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<1*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<2*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<2*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<2*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<3*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<3*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<3*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<4*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<4*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<4*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<5*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<5*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<5*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<6*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<6*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<6*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<7*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<7*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<7*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<8*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<8*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<8*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<9*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<9*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<9*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<10*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<10*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<10*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<11*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<11*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<11*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<12*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<12*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<12*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<13*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<13*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<13*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<14*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<14*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<14*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<15*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<15*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<15*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<16*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<16*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<16*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<17*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<17*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<17*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<18*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<18*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<18*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<19*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<19*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<19*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<20*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<20*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<20*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<21*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<21*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<21*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<22*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<22*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<22*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<23*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<23*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<23*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<24*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<24*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<24*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<25*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<25*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<25*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<26*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<26*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<26*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<27*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<27*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<27*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<28*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<28*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<28*3+2>(minVolRange,maxVolRange);
                gs.setInputRange<29*3+0>(minBtcRange,maxBtcRange);
                gs.setInputRange<29*3+1>(minVolRange,maxVolRange);
                gs.setInputRange<29*3+2>(minVolRange,maxVolRange);
                gs.setOutputRange<0>(minBtcRange,maxBtcRange);
            }
            gs.setStableMutChance(0.1);
            gs.train(nbEpochs,true);

            NetTopology bestNet = gs.getBestNetwork();
            NormalizedValue<float> btcVal(minBtcRange, maxBtcRange, -1, 1);
            std::cout << "Training finished - outputing test data along with expected and actual results of best agent" << std::endl;

            std::ofstream ofs("btc_usd_train_test_results.csv");

            ofs << std::setprecision(10) << std::fixed << "dataIdx,expectedOutput,networkOutput" << std::endl;
            for(size_t dataIdx = 0; dataIdx < gs.testData.data.size(); ++dataIdx)
            {
                auto& dataEntry = gs.testData.data[dataIdx];

                const auto& outputs = gs.processWithBestNetTrainDataAtIndex(dataIdx);

                ofs << std::setprecision(10) << std::fixed << dataIdx << "," << dataEntry.outputs[0] << "," << outputs[0] << std::endl;
            }

            JSONArchive jsonArchive("net01.json");
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


            gs.exportStatisticsToCSV("gdax_test_epoch_stats.csv");
            return 0;
        }


    }
}
