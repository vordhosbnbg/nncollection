#include <string>
#include <iostream>
#include <set>
#include <map>
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
            size_t ind = 0;
            for(char c : data)
            {
                mapped_data.emplace(c, ind++);
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

    size_t getCharIndex(char c)
    {
        size_t retVal = 0;
        auto it = mapped_data.find(c);
        if(it != mapped_data.end())
        {
            retVal = it->second;
        }

        return retVal;
    }

    std::set<char> data;
    std::map<char, size_t> mapped_data;
};

void analyse(const std::string& textData, const std::string& dictData)
{
    Dictionary dict;
    dict.read(textData);
    dict.write(dictData);
}


template <typename T, size_t inputSize>
void setValues(T vec, size_t offset, float value)
{
    for (size_t ind = 0; ind < inputSize; ++ind)
    {
        vec[offset+ind] = value;
    }
}

void train(const std::string& textData, const std::string& dictData, const std::string& netFilename)
{
    Dictionary dict;
    dict.read(dictData);

    constexpr size_t charsToLookBack = 20;

    using NetTopology = DynamicNetworkAdapter<78*charsToLookBack /*inputs*/,
                                  78 /*outputs*/,
                                  78*10 /*HL#2 neurons*/,
                                  78*5 /*HL#3 neurons*/>;
    using GenSim =  GeneticSimulation<
                    NetTopology,
                    80 /*agents*/,
                    1 /*keep best*/,
                    0 /*survival chance of rest*/>;

    std::unique_ptr<GenSim> gsPtr;

    std::random_device rd;
    std::minstd_rand re{rd()};
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

    GenSim& gs = *gsPtr.get();

    std::vector<char> text;
    std::ifstream ifs(textData);
    if(ifs.is_open())
    {
        char c;
        while(ifs.get(c))
        {
            if(c >= 32 && c <= 126)
            {
                text.emplace_back(c);
            }
        }
    }

    // prepare test data
    constexpr size_t trainOffsetMax = 2000;
    for(size_t idx = charsToLookBack; idx < trainOffsetMax; ++idx)
    {
        auto& entry = gs.testData.addEntry();
        size_t inputOffset = 0;
        for(int charIndexForEntry = charsToLookBack; charIndexForEntry >= 0; --charIndexForEntry)
        {
            // set all inputs for the character position to 0
            setValues<decltype(entry.inputs), charsToLookBack>(entry.inputs, inputOffset, 0.0f);
            size_t charIndex = dict.getCharIndex(text[idx-(size_t)charIndexForEntry]);
            entry.inputs[inputOffset+charIndex] = 1.0f;
            inputOffset += charsToLookBack;
        }

        size_t outputCharIndex = dict.getCharIndex(text[idx+1]);
        setValues<decltype(entry.outputs), 78>(entry.outputs, 0, 0.0f);
        entry.outputs[outputCharIndex] = 1.0f;
    }

    gs.train(nbEpochs,true);
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
