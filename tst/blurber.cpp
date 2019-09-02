#include <string>
#include <iostream>
#include <set>
#include <fstream>
#include "geneticsimulation.h"
#include "dnetworkadapter.h"
#include "jsonarchive.h"

void printHelp()
{
    std::cout << "Usage:\n"
                 "blurber analyse data.txt dict.dat - analyse the unique characters in the text\n"
                 "blurber train data.txt dict.dat [networkData.json] - train blurber on text using dictinary and optionaly an array of neural networks\n"
                 "blurber blurb dict.dat networkData.json networkIndex - use trained network to blurb endlessly\n";
}


struct Dictionary
{
    bool read(const std::string& filename)
    {
        std::ifstream ifs(filename);
        if(ifs.is_open())
        {
            char c;
            while(ifs.get(c))
            {
                if(c >= 32 && c <= 126)
                {
                    data.emplace(c);
                }
            }
        }
        return true;
    }


    bool write(const std::string& filename)
    {
        std::ofstream ofs(filename);
        for(char c : data)
        {
            ofs << c;
        }
        return true;
    }

    std::set<char> data;
};

void analyse(const std::string& textData, const std::string& dictData)
{
    Dictionary dict;
    dict.read(textData);
    dict.write(dictData);
}

void train(const std::string& textData, const std::string& dictData, const std::string& netFilename)
{
    Dictionary dict;
    dict.read(dictData);

    using NetTopology = DynamicNetworkAdapter<78*20 /*inputs*/,
                                  78 /*outputs*/,
                                  78*2 /*HL#1 neurons*/,
                                  78*3 /*HL#1 neurons*/,
                                  78*4 /*HL#1 neurons*/,
                                  78*3 /*HL#1 neurons*/,
                                  78*2 /*HL#1 neurons*/>;
    using GenSim =  GeneticSimulation<
                    NetTopology,
                    1000 /*agents*/,
                    500 /*keep best*/,
                    50 /*survival chance of rest*/>;

    std::unique_ptr<GenSim> gsPtr;

    std::random_device rd;
    std::mt19937 re{rd()};
    std::uniform_real_distribution<float> simpleDist(-3.14,3.14);
    constexpr unsigned int nbEpochs = 7500;
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

}

void blurb(const std::string& dictData, const std::string& networkData, const std::string& networkIndex)
{

}

int main (int argc, char* argv[])
{
    if(argc >= 2 && argc <= 6)
    {
        std::string arg1(argv[1]);
        std::string arg2(argc > 2 ? argv[2] : "");
        std::string arg3(argc > 3 ? argv[3] : "");
        std::string arg4(argc > 4 ? argv[4] : "");
        std::string arg5(argc > 5 ? argv[5] : "");

        if(arg1 == "help")
        {
            printHelp();
        }
        else if(arg1 == "analyse")
        {
            if(!arg2.empty() && !arg3.empty())
            {
                analyse(arg2, arg3);
            }
        }
        else if(arg1 == "train")
        {
            if(!arg2.empty() && !arg3.empty())
            {
                train(arg2, arg3, arg4);
            }
        }
        else if(arg1 == "blurb")
        {
            if(!arg2.empty() && !arg3.empty())
            {
                blurb(arg2, arg3, arg4);
            }
        }
        else
        {
            printHelp();
        }

    }
    else
    {
        printHelp();
    }

    return 0;
}
