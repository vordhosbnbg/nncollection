#include "ffnetwork.h"
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
}
