#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cstring>
#include <cmath>
#include <functional>
#include <numeric>
#include <algorithm>
#include <utility>

using namespace std;

const int MAXN = 1024;

// input
int N;

class Cue;
class Transition;

class State
{
public:
    string name;
    double reward;
    Cue *cue;
    vector<Transition*> in, out;

    State() :
        name(""),
        reward(0),
        cue(NULL)
    {}
};

class Cue
{
public:
    string name;
    double value; // what is the expected reward for this cue -- this could be deduced from the graph, in theory
    vector<State*> states;

    Cue() :
        name(""),
        value(0)
    { }
};

class Transition
{
public:
    string name;
    State *from;
    State *to;

    Transition() :
        name(""),
        from(NULL),
        to(NULL)
    { }
};

class Action : Transition
{
};

class Chance : Transition
{
    double probability;

    Chance() :
        Transition(),
        probability(0)
    { }
};

class ExperimentalModel
{
    vector<State*> states;
    vector<Transition*> transitions;
    vector<Cue*> cues;
};

int main()
{
    return 0;
}
