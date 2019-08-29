#include "ffnetwork.h"
#include "dynamicnetwork.h"
#include "jsonarchive.h"


int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
    std::random_device rd;
    std::mt19937 re{rd()};
    FFNetwork<5,1,4,3,2> net1(re);

    JSONArchive jsonArchive("net1.json");
    jsonArchive.write(net1);

    FFNetwork<5,1,4,3,2> net2(re);

    JSONArchive jsonArchive2("net1.json");
    jsonArchive2.read(net2);

    JSONArchive jsonArchive3("net2.json");
    jsonArchive3.write(net2);

    DynamicNetwork dnet1;
    dnet1.setInputNb(2);
    dnet1.setOutputNb(2);
    dnet1.setHiddenLayers(3u,4u,3u);
    dnet1.connectNetwork();

    JSONArchive jsonArchiveDnet1("dnet1.json");
    jsonArchiveDnet1.write(dnet1);
    JSONArchive jsonArchiveDnet2("dnet1.json");
    DynamicNetwork dnet2;
    jsonArchiveDnet2.read(dnet2);
    JSONArchive jsonArchiveDnet3("dnet2.json");
    jsonArchiveDnet3.write(dnet2);

}
