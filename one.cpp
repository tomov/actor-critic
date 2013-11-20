#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cstring>
#include <cmath>

using namespace std;

const int MAXN = 1024;

// input
int N;

struct transition_t
{
    int from;
    int to;
    string name;
    double prob; // 1 for actions

    // learning variables
    double H;
    double policy;

    // record
    double delta_avg;
    int times;
};

enum state_type_t {CHOICE, PROBABILISTIC};

struct state_t
{
    // input variables
    int idx;
    string name;
    state_type_t type;
    double R;
    vector<transition_t> next;

    // learning variables
    double V;
};

state_t state[MAXN];

// learning params
double eta = 0.05; // critic learning rate
double alpha = 0.05; // actor learning rate
double discount = 0.99;   // discount
double beta = 0.01; // softmax temperature
double min_R = 1; // minimum reward for probability matching

enum method_t {SOFTMAX, MATCHING_NOT_WORKING, MATCHING};
method_t method = MATCHING;

double get_action_weight(state_t &S, int i)
{
    transition_t &trans = S.next[i];
    switch(method)
    {
        case SOFTMAX:
        {
            return exp(beta * trans.H);
        }
        break;
        case MATCHING_NOT_WORKING:
        {
            return max(trans.H, min_R);
        }
        break;
        case MATCHING:
        {
            state_t &Snew = state[trans.to];
            return max(Snew.V, min_R);
        }
        break;
        default:
        {
            return -1;
        }
    }
}

void get_policy(state_t &S)
{
    if (S.type != CHOICE)
    {
        // there is no policy for non-choice transitions (i.e. non-actions)
        return;
    }
    double total = 0;
    for (int i = 0; i < S.next.size(); i++)
    {
        double weight = get_action_weight(S, i);
        S.next[i].policy = weight;
        total += weight;
    }
    for (int i = 0; i < S.next.size(); i++)
    {
        S.next[i].policy /= total;
    }
}

int pick_next(state_t &S)
{
    double r = (double)rand() / RAND_MAX;
    double tot = 0;
    for (int i = 0; i < S.next.size(); i++)
    {
        transition_t &trans = S.next[i];
        if (S.type == CHOICE)
        {
            tot += trans.policy;
        }
        else
        {
            tot += trans.prob;
        }
        if (tot >= r)
        {
            return i;
        }
    }
    return -1; // no transition available -- terminal state
}

void read()
{
    cin>>N;
    string type;
    for (int i = 0; i < N; i++)
    {
        state_t &S = state[i];
        S.idx = i;
        cin>>S.R>>S.name>>type;
        if (type[0] == 'C')
        {
            S.type = CHOICE;
        }
        else
        {
            S.type = PROBABILISTIC;
        }
    }
    transition_t trans;
    while (cin>>trans.from>>trans.to>>trans.name)
    {
        state_t &S = state[trans.from];
        if (S.type == CHOICE)
        {
            trans.prob = 1; // actions are deterministic
        }
        else
        {
            cin>>trans.prob;
        }
        trans.H = 0;
        trans.policy = 0;
        S.next.push_back(trans);
    }
    for (int i = 0; i < N; i++)
    {
        get_policy(state[i]);
    }
}

void print()
{
    for (int i = 0; i < N; i++)
    {
        state_t &S = state[i];
        string type;
        if (S.type == CHOICE)
        {
            type = "choice";
        }
        else
        {
            type = "probabilistic";
        }
        cout<<"From "<<S.name<<" (R = $"<<S.R<<", V = "<<S.V<<") -- "<<type<<":\n";
        for (int j = 0; j < S.next.size(); j++)
        {
            transition_t &trans = S.next[j];
            state_t &Snew = state[trans.to];
            if (S.type == CHOICE)
            {
                cout<<"                                                      "<<trans.name<<" (action) to "<<Snew.name<<" (H = "<<trans.H<<", policy = "<<trans.policy<<", chosen "<<trans.times<<" times, PE = "<<trans.delta_avg<<")\n";
            }
            else
            {
                cout<<"                                                      "<<trans.name<<" (prob) to "<<Snew.name<<" (H = "<<trans.H<<", prob = "<<trans.prob<<", chosen "<<trans.times<<" times, PE = "<<trans.delta_avg<<")\n";
            }
        }
    }
}

void trial(bool do_print)
{
    int idx = 0;
    if (do_print) cout<<"\n  ---------------------- TRIAL --------------\n\n";
    int count = 0;
    double bias = 0;
    double delta_prev = 0;
    double delta_prevprev = 0;
    while (true)
    {
        state_t &S = state[idx];
        int a = pick_next(S);
        if (a == -1)
        {
            break;
        }
        transition_t &trans = S.next[a];
        state_t &Snew = state[trans.to];

        double R = Snew.R;
        double delta = R + discount * Snew.V - S.V;
        S.V += eta * delta;

        trans.H += alpha * delta;
        get_policy(S);

        double add_what = delta + delta_prev + delta_prevprev + bias;
        trans.delta_avg = (trans.delta_avg * trans.times + add_what) / (trans.times + 1);
        trans.times++;

        if (do_print) cout<<" from "<<S.name<<", "<<trans.name<<" --> "<<Snew.name<<", PE = "<<delta<<"\n";

        idx = trans.to;
        delta_prevprev = delta_prev;
        delta_prev = delta;
    }

    if (do_print) cout<<"\n";
    if (do_print) print();
}

void learn()
{
    for (int iter = 0; iter < 300000; iter++)
    {
        trial(false);
    }
}

int main()
{
    srand(123);

    read();

    learn();

    cout<<" \n\n        -------------------- FINAL VALUES AND POLICY --------------- \n\n\n";
    print();
    
    return 0;
}
