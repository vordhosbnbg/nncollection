#include <cmath>
#include <map>
#include <fstream>
#include <iomanip>
#include "geneticsimulation.h"
#include "ffnetwork.h"
#include "jsonarchive.h"

class CSVRow
{
    public:
        std::string const& operator[](std::size_t index) const
        {
            return m_data[index];
        }
        std::size_t size() const
        {
            return m_data.size();
        }
        void readNextRow(std::istream& str)
        {
            std::string         line;
            std::getline(str, line);

            std::stringstream   lineStream(line);
            std::string         cell;

            m_data.clear();
            while(std::getline(lineStream, cell, ','))
            {
                m_data.push_back(cell);
            }
            // This checks for a trailing comma with no data after it.
            if (!lineStream && cell.empty())
            {
                // If there was a trailing comma then add an empty element.
                m_data.push_back("");
            }
        }
    private:
        std::vector<std::string>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
}


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
            std::mt19937 re{rd()};
            std::uniform_real_distribution<float> simpleDist(-3.14,3.14);
            using NetTopology = FFNetwork<90 /*inputs*/,1 /*outputs*/,32 /*HL#1 neurons*/,32 /*HL#1 neurons*/,32 /*HL#1 neurons*/>;
            using GenSim =  GeneticSimulation<
                            NetTopology,
                            1000 /*agents*/,
                            100 /*keep best*/,
                            5 /*survival chance of rest*/>;

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
            constexpr unsigned int nbEpochs= 100;
            // prepare test data - sinf() function
            for(size_t idx = 30; idx < btcData.size(); ++idx)
            {
                auto& entry = gs.testData.addEntry();
                for(int daysBack = 30; daysBack > 0; --daysBack)
                {
                    entry.inputs[0+30-daysBack] = btcData[idx-daysBack].price;
                    entry.inputs[1+30-daysBack] = btcData[idx-daysBack].volumeFrom;
                    entry.inputs[2+30-daysBack] = btcData[idx-daysBack].volumeTo;
                }
                entry.outputs[0] = btcData[idx].price;
            }

            constexpr float minBtcRange = 0;
            constexpr float maxBtcRange = 30000;
            constexpr float minVolRange = 0;
            constexpr float maxVolRange = 1300000000;
            {
                gs.setInputRange<0+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<0+1>(-minVolRange,maxVolRange);
                gs.setInputRange<0+2>(-minVolRange,maxVolRange);
                gs.setInputRange<1+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<1+1>(-minVolRange,maxVolRange);
                gs.setInputRange<1+2>(-minVolRange,maxVolRange);
                gs.setInputRange<2+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<2+1>(-minVolRange,maxVolRange);
                gs.setInputRange<2+2>(-minVolRange,maxVolRange);
                gs.setInputRange<3+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<3+1>(-minVolRange,maxVolRange);
                gs.setInputRange<3+2>(-minVolRange,maxVolRange);
                gs.setInputRange<4+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<4+1>(-minVolRange,maxVolRange);
                gs.setInputRange<4+2>(-minVolRange,maxVolRange);
                gs.setInputRange<5+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<5+1>(-minVolRange,maxVolRange);
                gs.setInputRange<5+2>(-minVolRange,maxVolRange);
                gs.setInputRange<6+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<6+1>(-minVolRange,maxVolRange);
                gs.setInputRange<6+2>(-minVolRange,maxVolRange);
                gs.setInputRange<7+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<7+1>(-minVolRange,maxVolRange);
                gs.setInputRange<7+2>(-minVolRange,maxVolRange);
                gs.setInputRange<8+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<8+1>(-minVolRange,maxVolRange);
                gs.setInputRange<8+2>(-minVolRange,maxVolRange);
                gs.setInputRange<9+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<9+1>(-minVolRange,maxVolRange);
                gs.setInputRange<9+2>(-minVolRange,maxVolRange);
                gs.setInputRange<10+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<10+1>(-minVolRange,maxVolRange);
                gs.setInputRange<10+2>(-minVolRange,maxVolRange);
                gs.setInputRange<11+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<11+1>(-minVolRange,maxVolRange);
                gs.setInputRange<11+2>(-minVolRange,maxVolRange);
                gs.setInputRange<12+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<12+1>(-minVolRange,maxVolRange);
                gs.setInputRange<12+2>(-minVolRange,maxVolRange);
                gs.setInputRange<13+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<13+1>(-minVolRange,maxVolRange);
                gs.setInputRange<13+2>(-minVolRange,maxVolRange);
                gs.setInputRange<14+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<14+1>(-minVolRange,maxVolRange);
                gs.setInputRange<14+2>(-minVolRange,maxVolRange);
                gs.setInputRange<15+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<15+1>(-minVolRange,maxVolRange);
                gs.setInputRange<15+2>(-minVolRange,maxVolRange);
                gs.setInputRange<16+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<16+1>(-minVolRange,maxVolRange);
                gs.setInputRange<16+2>(-minVolRange,maxVolRange);
                gs.setInputRange<17+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<17+1>(-minVolRange,maxVolRange);
                gs.setInputRange<17+2>(-minVolRange,maxVolRange);
                gs.setInputRange<18+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<18+1>(-minVolRange,maxVolRange);
                gs.setInputRange<18+2>(-minVolRange,maxVolRange);
                gs.setInputRange<19+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<19+1>(-minVolRange,maxVolRange);
                gs.setInputRange<19+2>(-minVolRange,maxVolRange);
                gs.setInputRange<20+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<20+1>(-minVolRange,maxVolRange);
                gs.setInputRange<20+2>(-minVolRange,maxVolRange);
                gs.setInputRange<21+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<21+1>(-minVolRange,maxVolRange);
                gs.setInputRange<21+2>(-minVolRange,maxVolRange);
                gs.setInputRange<22+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<22+1>(-minVolRange,maxVolRange);
                gs.setInputRange<22+2>(-minVolRange,maxVolRange);
                gs.setInputRange<23+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<23+1>(-minVolRange,maxVolRange);
                gs.setInputRange<23+2>(-minVolRange,maxVolRange);
                gs.setInputRange<24+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<24+1>(-minVolRange,maxVolRange);
                gs.setInputRange<24+2>(-minVolRange,maxVolRange);
                gs.setInputRange<25+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<25+1>(-minVolRange,maxVolRange);
                gs.setInputRange<25+2>(-minVolRange,maxVolRange);
                gs.setInputRange<26+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<26+1>(-minVolRange,maxVolRange);
                gs.setInputRange<26+2>(-minVolRange,maxVolRange);
                gs.setInputRange<27+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<27+1>(-minVolRange,maxVolRange);
                gs.setInputRange<27+2>(-minVolRange,maxVolRange);
                gs.setInputRange<28+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<28+1>(-minVolRange,maxVolRange);
                gs.setInputRange<28+2>(-minVolRange,maxVolRange);
                gs.setInputRange<29+0>(-minBtcRange,maxBtcRange);
                gs.setInputRange<29+1>(-minVolRange,maxVolRange);
                gs.setInputRange<29+2>(-minVolRange,maxVolRange);
                gs.setOutputRange<0>(minBtcRange,maxBtcRange);
            }

            gs.train(nbEpochs,true);

            NetTopology bestNet = gs.getBestNetwork();
            NormalizedValue<float> btcVal(minBtcRange, maxBtcRange, -1, 1);
            std::cout << "Training finished - outputing test data along with expected and actual results of best agent" << std::endl;

//            for(auto& entry : gs.testData)
//            {

//            }
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
