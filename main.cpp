#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "model.h"

int main()
{
    ExperimentalModel model;

    model.read();
    model.print();
    return 0;
}
