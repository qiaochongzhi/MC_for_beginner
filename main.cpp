#include "MonteCarlo.h"
#include "linkList.h"

int main()
{
    mc = MonteCarlo(64, 3, 1.0, 2.5, true);
    mc.SetBox({10., 10., 10.});
    mc.InitPosition();
    return 0;
}