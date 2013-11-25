#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <map>
#include <sstream>

using namespace std;

class Cue;
class Transition;

enum StateType
{
    PROBABILISTIC,
    DETERMINISTIC,
};


class State
{
public:
    string name;
    double reward;
    Cue *cue;
    vector<Transition*> in, out;
    StateType type;
    string extra;

    State() :
        name(""),
        reward(0),
        cue(NULL),
        type(PROBABILISTIC),
        extra("")
    { }
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


enum TransitionType
{
    CHANCE,
    CHOICE
};


class Transition
{
public:
    State *from;
    State *to;

    Transition() :
        from(NULL),
        to(NULL)
    { }

    virtual string GetExtraString() = 0;
};


class Chance : public Transition
{
public:
    double probability;

    Chance() :
        Transition(),
        probability(0)
    { }

    string GetExtraString()
    {
       ostringstream ss;
       ss<<"prob = "<<probability;
       return ss.str();
    }
};


class Choice : public Transition
{
public:
    string name;

    Choice() :
        Transition(),
        name("")
    { }

    string GetExtraString()
    {
        ostringstream ss;
        ss<<"ACTION: "<<name;
        return ss.str();
    }
};


class ExperimentalModel
{
public:
    vector<State*> states;
    vector<Transition*> transitions;
    vector<Cue*> cues;

    map<string, State*> state_from_name;
    map<string, Transition*> transition_from_name;
    map<string, Cue*> cue_from_name;

    State* start;
    State* end;

    void Read()
    {
        int C;
        cin>>C;
        for (int i = 0; i < C; i++)
        {
            Cue *cue = new Cue();
            cin>>cue->name>>cue->value;
            cues.push_back(cue);
            if (cue_from_name.find(cue->name) != cue_from_name.end())
            {
                cerr<<"Duplicate cue name '"<<cue->name<<"'. Aborting...\n";
                exit(0);
            }
            cue_from_name[cue->name] = cue;
        }

        int N;
        cin>>N;
        for (int i = 0; i < N; i++)
        {
            State *state = new State();
            string type;
            string cue_name;
            cin>>state->name>>state->reward>>type>>cue_name>>state->extra;
            if (type[0] == 'D' or type[0] == 'd')
            {
                state->type = DETERMINISTIC;
            }
            else
            {
                state->type = PROBABILISTIC;
            }
            if (cue_from_name.find(cue_name) != cue_from_name.end())
            {
                Cue* cue = cue_from_name[cue_name];
                state->cue = cue;
                cue->states.push_back(state);
            }
            states.push_back(state);
            if (state_from_name.find(state->name) != state_from_name.end())
            {
                cerr<<"Duplicate state name '"<<state->name<<"'. Aborting...\n";
                exit(0);
            }
            state_from_name[state->name] = state;
            cout<<state->name<<" "<<state->reward<<" "<<state->type<<" "<<cue_name<<"\n";
        }

        string from_name, to_name;
        while (cin>>from_name>>to_name)
        {
            Transition *trans;
            if (state_from_name.find(from_name) == state_from_name.end())
            {
                cerr<<"No state with name '"<<from_name<<"' exists. Aborting...\n";
                exit(0);
            }
            if (state_from_name.find(to_name) == state_from_name.end())
            {
                cerr<<"No state with name '"<<to_name<<"' exists. Aborting...\n";
                exit(0);
            }
            if (state_from_name[from_name]->type == PROBABILISTIC)
            {
                Chance *chance = new Chance();
                chance->from = state_from_name[from_name];
                chance->to = state_from_name[to_name];
                cin>>chance->probability;
                trans = chance;
            }
            else
            {
                Choice *choice = new Choice();
                choice->from = state_from_name[from_name];
                choice->to = state_from_name[to_name];
                cin>>choice->name;
                trans = choice;
            }
            transitions.push_back(trans);
            trans->from->out.push_back(trans);
            trans->to->in.push_back(trans);
        }

        for (int i = 0; i < states.size(); i++)
        {
            State *state = states[i];
            if (state->in.size() == 0)
            {
                start = state;
            }
            if (state->out.size() == 0)
            {
                end = state;
            }
        }
    }


    void Print()
    {
        cout<<" Cues:\n";
        for (int i = 0; i < cues.size(); i++)
        {
            Cue* cue = cues[i];
            cout<<"   "<<cue->name<<"  --> $"<<cue->value<<". States: ";
            for (int j = 0; j < cue->states.size(); j++)
            {
                cout<<cue->states[j]->name<<", ";
            }
            cout<<"\n";
        }
        cout<<"\n States:\n";
        for (int i = 0; i < states.size(); i++)
        {
            State *state = states[i];
            cout<<"   "<<state->name<<" --> $"<<state->reward<<". Cue: "<<(state->cue ? state->cue->name : "no-cue")<<". Type: "<<(state->type == PROBABILISTIC ? "probabilistic" : "DETERMINISTIC")<<", extra = "<<state->extra<<"\n";
            for (int j = 0; j < state->out.size(); j++)
            {
                Transition* trans = state->out[j];
                cout<<"                                                               "<<trans->from->name<<" "<<trans->to->name<<" ("<<trans->GetExtraString()<<")\n";
            }
        }
        cout<<"\n";
    }


    ExperimentalModel() :
        start(NULL),
        end(NULL)
    { }


    ~ExperimentalModel()
    {
        for (int i = 0; i < states.size(); i++)
        {
            delete states[i];
        }
        for (int i = 0; i < transitions.size(); i++)
        {
            delete transitions[i];
        }
        for (int i = 0; i < cues.size(); i++)
        {
            delete cues[i];
        }
    }
};

#endif
