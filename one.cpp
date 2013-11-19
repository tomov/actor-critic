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
vector<int> adj[MAXN];
double reward[MAXN];

// learning variables
double V[MAXN];
vector<double> policy[MAXN];
vector<double> H[MAXN];

// for visualization
vector<string> actions[MAXN];
string states[MAXN];

// learning params
double eta = 0.1; // critic learning rate
double alpha = 0.05; // actor learning rate
double discount = 0.99;   // discount
double beta = 0.01; // softmax temperature
double min_R = 1; // minimum reward for probability matching

enum method_t {SOFTMAX, MATCHING_NOT_WORKING, MATCHING};
method_t method = MATCHING;

double get_action_weight(int state, int action)
{
    switch(method)
    {
        case SOFTMAX:
        {
            return exp(beta * H[state][action]);
        }
        break;
        case MATCHING_NOT_WORKING:
        {
            return max(H[state][action], min_R);
        }
        break;
        case MATCHING:
        {
            int next = adj[state][action];
            double value = V[next];
            return max(value, min_R);
        }
        break;
        default:
        {
            return -1;
        }
    }
}

void get_policy(int state)
{
    double total = 0;
    for (int i = 0; i < adj[state].size(); i++)
    {
        double weight = get_action_weight(state, i);
        policy[state][i] = weight;
        total += weight;
    }
    for (int i = 0; i < adj[state].size(); i++)
    {
        policy[state][i] /= total;
    }
}

int pick_action(int state)
{
    double r = (double)rand() / RAND_MAX;
    double tot = 0;
    for (int i = 0; i < policy[state].size(); i++)
    {
        tot += policy[state][i];
        if (tot >= r)
        {
            return i;
        }
    }
    return -1; // no action available -- terminal state
}

void read()
{
    cin>>N;
    for (int i = 0; i < N; i++)
    {
        cin>>reward[i]>>states[i];
    }
    int a, b;
    string s;
    while (cin>>a>>b)
    {
        getline(cin, s);
        actions[a].push_back(s);
        adj[a].push_back(b);
        H[a].push_back(0); // Softmax
        policy[a].push_back(0);
    }
    for (int i = 0; i < N; i++)
    {
        get_policy(i);
    }
}

void print()
{
    for (int i = 0; i < N; i++)
    {
        cout<<"From "<<states[i]<<" (R = $"<<reward[i]<<", V = "<<V[i]<<") to:\n";
        for (int j = 0; j < adj[i].size(); j++)
        {
            int next = adj[i][j];
            cout<<"                                                       "<<states[next]<<":"<<actions[i][j]<<" (H = "<<H[i][j]<<", prob = "<<policy[i][j]<<")\n";
        }
    }
}

void trial()
{
    int S = 0;
    cout<<"\n  ---------------------- TRIAL --------------\n\n";
    int count = 0;
    while (true)
    {
        int a = pick_action(S);
        if (a == -1)
        {
            break;
        }
        int Snew = adj[S][a];

        double R = reward[Snew];
        double delta = R + discount * V[Snew] - V[S];
        V[S] += eta * delta;

        H[S][a] += alpha * delta;
        get_policy(S);

        cout<<" from "<<states[S]<<","<<actions[S][a]<<" --> "<<states[Snew]<<", PE = "<<delta<<"\n";

        S = Snew;
    }

    cout<<"\n";
    print();
}

void learn()
{
    for (int iter = 0; iter < 3000; iter++)
    {
        trial();
    }
}

int main()
{
    srand(123);

    read();
    print();

    learn();
    
    return 0;
}
