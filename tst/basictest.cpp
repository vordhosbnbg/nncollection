#include "ffnetwork.h"


int main ([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
    FFNetwork<5,1,4,3,2> net;
    net.process();
    return 0;
}
