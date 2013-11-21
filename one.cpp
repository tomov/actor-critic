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

    transition_t() :
        from(0),
        to(0),
        name(""),
        prob(0),
        H(0),
        policy(0),
        delta_avg(0),
        times(0)
    { }
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

    // cue variables
    bool is_cue;
    char cue_id;

    state_t() :
        idx(0),
        name(""),
        type(CHOICE),
        R(0),
        is_cue(0),
        cue_id(0)
    { }
};

state_t state[MAXN];

struct cue_t
{
    int state_idx;
    int seen;
    int rewarded;
    double exp_reward_hardcoded;

    cue_t() :
        state_idx(0),
        seen(0),
        rewarded(0),
        exp_reward_hardcoded(0)
    { }
};

cue_t cue[256];

// learning params
double eta = 0.01; // critic learning rate
double alpha = 0.01; // actor learning rate
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

        if (S.name.length() == 5 && S.name.substr(0, 3) == "cue")
        {
            S.is_cue = true;
            S.cue_id = S.name[4];
            cue[S.cue_id].state_idx = i;
            // TODO #hardcoded... hack FIXME
            cue[S.cue_id].exp_reward_hardcoded = (S.cue_id - 'A' + 1) * 25;
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
    double bias = 80; // TODO sketchy... #hardcoded
    double delta_prev = 0;
    double delta_prevprev = 0;
    char cue_id = 0;
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

        // bookkeeping

        double add_what = delta + delta_prev + delta_prevprev + bias;
        trans.delta_avg = (trans.delta_avg * trans.times + add_what) / (trans.times + 1);
        trans.times++;

        if (S.is_cue)
        {
            // this is our new cue!
            cue_id = S.cue_id;
            cue[cue_id].seen++;
        }

        if (S.R > 0)
        {
            // this cue got rewarded
            cue[cue_id].rewarded++;
        }

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



void print_2a(bool matlab = true)
{
    cout<<"\n %%---- figure 2a -----\n\n";
    if (matlab) cout<<"y2a = [";
    for (char cue_id = 'A'; cue_id <= 'D'; cue_id++)
    {
        double R = (double)cue[cue_id].rewarded / cue[cue_id].seen;
        if (!matlab) cout<<"  cue "<<cue_id<<" rewarded "<<cue[cue_id].rewarded<<" out of "<<cue[cue_id].seen<<" times = "<<R<<"\n";
        if (matlab) cout<<R<<", "<<R<<"; ";
    }
    if (matlab) cout<<"]\n";
    if (matlab)
    {
        cout<<"subplot(2, 2, 1);\n";
        cout<<"bar([25, 50, 75, 100], y2a);\n";
        cout<<"xlabel('Reward probability');\n";
        cout<<"ylabel('Obtained reward (R) (%)');\n";
    }
    cout<<"\n";
}



void print_2b(bool matlab = true)
{
     cout<<"\n %%---- figure 2b -----\n\n";  
     if (matlab) cout<<"x2b = [";
     for (int i = 0; i < N; i++)
     {
         state_t &S = state[i];
         if (S.name.length() == 5 && S.name.substr(0, 2) == "go")
         {
             char left_cue_id = S.name[3];
             char right_cue_id = S.name[4];
             double Rleft = (double)cue[left_cue_id].rewarded / cue[left_cue_id].seen;
             double Rright = (double)cue[right_cue_id].rewarded / cue[right_cue_id].seen;

             double Pright = Rright / (Rright + Rleft);
             if (!matlab) cout<<" Rright / (Rright + Rleft) = "<<Rright<<" / ("<<Rright<<" + "<<Rleft<<") = "<<Pright<<"\n";
             if (matlab) cout<<Pright<<", ";

             double Pleft = Rleft / (Rright + Rleft);
             if (!matlab) cout<<" Rright / (Rright + Rleft) = "<<Rleft<<" / ("<<Rleft<<" + "<<Rright<<") = "<<Pleft<<"\n";
             if (matlab) cout<<Pleft<<", ";
         }
     }
     if (matlab) cout<<"]\n";

     if (matlab) cout<<"y2b = [";
     for (int i = 0; i < N; i++)
     {
         state_t &S = state[i];
         if (S.name.length() == 5 && S.name.substr(0, 2) == "go")
         {
             transition_t &left_trans = S.next[0];
             transition_t &right_trans = S.next[1];

             double Cright = (double)right_trans.times / (left_trans.times + right_trans.times);
             if (!matlab) cout<<" Cright ------>>>> "<<Cright<<"\n";
             if (matlab) cout<<Cright<<", ";

             double Cleft = (double)left_trans.times / (left_trans.times + right_trans.times);
             if (!matlab) cout<<" Cright ------>>>> "<<Cleft<<"\n";
             if (matlab) cout<<Cleft<<", ";
         }
     }
     if (matlab) cout<<"]\n";

     if (matlab)
     {
         cout<<"subplot(2, 2, 2);\n";
         cout<<"scatter(x2b, y2b);\n";
         cout<<"xlabel('R_{right} / (R_{right} + R_{left})');\n";
         cout<<"ylabel('C_{right}');\n";
         cout<<"axis([0 1 0 1]);\n";
         cout<<"lsline;\n";
     }
     cout<<"\n";
}



void print_2c(bool matlab = true)
{
    cout<<"\n %%---- figure 2c ----\n\n";
    if (matlab) cout<<"y2c = [";
    for (int i = 0; i < N; i++)
    {
        state_t &S = state[i];
        if (S.name.length() == 4 && S.name.substr(0, 2) == "go") // TODO ALTERNATIVELY, measure PE after cue trials to account for negative effets of wrong choices (smaller PE)
        {
            char cue_id = S.name[3];
            transition_t &correct = S.next[0];
            if (!matlab) cout<<" cue "<<cue_id<<" average spike = "<<correct.delta_avg<<"\n";
            if (matlab) cout<<correct.delta_avg<<", "<<correct.delta_avg<<"; ";
        }
    }
    if (matlab) cout<<"]\n";
    if (matlab)
    {
        cout<<"subplot(2, 2, 3);\n";
        cout<<"bar([25, 50, 75, 100], y2c);\n";
        cout<<"xlabel('Reward probability');\n";
        cout<<"ylabel('PE ~ DA response rate (D)');\n";
    }
    cout<<"\n";
}



void print_2d(bool matlab = true)
{
     cout<<"\n %%---- figure 2d -----\n\n";  
     if (matlab) cout<<"x2b = [";
     for (int i = 0; i < N; i++)
     {
         state_t &S = state[i];
         if (S.name.length() == 5 && S.name.substr(0, 2) == "go")
         {
             transition_t &left_trans = S.next[0];
             transition_t &right_trans = S.next[1];
             double Dleft = left_trans.delta_avg; 
             double Dright = right_trans.delta_avg; 

             double Pright = Dright / (Dright + Dleft);
             if (!matlab) cout<<" Dright / (Dright + Dleft) = "<<Dright<<" / ("<<Dright<<" + "<<Dleft<<") = "<<Pright<<"\n";
             if (matlab) cout<<Pright<<", ";

             double Pleft = Dleft / (Dright + Dleft);
             if (!matlab) cout<<" Dright / (Dright + Dleft) = "<<Dleft<<" / ("<<Dleft<<" + "<<Dright<<") = "<<Pleft<<"\n";
             cout<<Pleft<<", ";
         }
     }
     if (matlab) cout<<"]\n";

     if (matlab) cout<<"y2b = [";
     for (int i = 0; i < N; i++)
     {
         state_t &S = state[i];
         if (S.name.length() == 5 && S.name.substr(0, 2) == "go")
         {
             transition_t &left_trans = S.next[0];
             transition_t &right_trans = S.next[1];

             double Cright = (double)right_trans.times / (left_trans.times + right_trans.times);
             if (!matlab) cout<<" Cright ------>>>> "<<Cright<<"\n";
             if (matlab) cout<<Cright<<", ";

             double Cleft = (double)left_trans.times / (left_trans.times + right_trans.times);
             if (!matlab) cout<<" Cright ------>>>> "<<Cleft<<"\n";
             if (matlab) cout<<Cleft<<", ";
         }
     }
     if (matlab) cout<<"]\n";

     if (matlab)
     {
         cout<<"subplot(2, 2, 4);\n";
         cout<<"scatter(x2b, y2b);\n";
         cout<<"xlabel('D_{right} / (D_{right} + D_{left})');\n";
         cout<<"ylabel('C_{right}');\n";
         cout<<"axis([0 1 0 1]);\n";
         cout<<"lsline;\n";
     }

     cout<<"\n";
}



void print_4ab(bool matlab = true)
{
     vector<pair<double, pair<double, pair<int, int> > > > Ds_a;
     vector<pair<double, pair<pair<double, double>, pair<int, int> > > > Ds_b;
     for (int i = 0; i < N; i++)
     {
         state_t &S = state[i];
         if (S.name.length() == 5 && S.name.substr(0, 2) == "go")
         {
             char low_cue_id = S.name[3];
             char high_cue_id = S.name[4];
             transition_t &low_trans = S.next[0];
             transition_t &high_trans = S.next[1];
             double Dlow = low_trans.delta_avg; 
             double Dhigh = high_trans.delta_avg; 

             double Davg = (Dlow * low_trans.times + Dhigh * high_trans.times) / (low_trans.times + high_trans.times);
             double expRlow = cue[low_cue_id].exp_reward_hardcoded;
             double expRhigh = cue[high_cue_id].exp_reward_hardcoded;
             double expR = (expRlow + expRhigh) / 2;  // TODO OMG WTF INVESTIGATE WHY DO THEY are averaging that way.... and not probability matching, which is how the monkey will see it
             Ds_a.push_back(make_pair(expR, make_pair(Davg, make_pair(expRhigh, expRlow))));
             if (low_cue_id != high_cue_id)
             {
                 Ds_b.push_back(make_pair(expR, make_pair(make_pair(Dhigh, Dlow), make_pair(expRhigh, expRlow))));
             }
         }
     }
     sort(Ds_a.begin(), Ds_a.end());

     cout<<"\n %%---- figure 4a ----\n\n";
     if (matlab)
     {
         cout<<"x4a = {";
         for (int i = 0; i < Ds_a.size(); i++)
         {
             cout<<"'"<<Ds_a[i].second.second.first<<"-"<<Ds_a[i].second.second.second<<"', ";
         }
         cout<<"}\n";
         cout<<"y4a = [";
         for (int i = 0; i < Ds_a.size(); i++)
         {
             cout<<Ds_a[i].second.first<<", ";
         }
         cout<<"]\n";
     }
     if (matlab)
     {
         cout<<"subplot(3, 2, 1);\n";
         cout<<"bar(y4a);\n";
         cout<<"set(gca,'XTickLabel',x4a);\n";
         cout<<"xlabel('State (pair)');\n";
         cout<<"ylabel('PE ~ DA response');\n";
     }
     cout<<"\n";

     cout<<"\n %%---- figure 4b ----\n\n";
     if (matlab)
     {
         cout<<"x4a = {";
         for (int i = 0; i < Ds_b.size(); i++)
         {
             cout<<"'"<<Ds_b[i].second.second.first<<"-"<<Ds_b[i].second.second.second<<"', ";
         }
         cout<<"}\n";
         cout<<"y4a = [";
         for (int i = 0; i < Ds_b.size(); i++)
         {
             cout<<Ds_b[i].second.first.first<<", "<<Ds_b[i].second.first.second<<"; ";
         }
         cout<<"]\n";
     }
     if (matlab)
     {
         cout<<"subplot(3, 2, 3);\n";
         cout<<"bar(y4a);\n";
         cout<<"set(gca,'XTickLabel',x4a);\n";
         cout<<"xlabel('State (pair)');\n";
         cout<<"ylabel('PE ~ DA response');\n";
     }
     cout<<"\n";
}


int main()
{
    srand(123);

    read();

    learn();

    cout<<" \n\n        -------------------- FINAL VALUES AND POLICY --------------- \n\n\n";
    print();

    print_2a();
    print_2b();
    print_2c();
    print_2d();

    print_4ab();

    
    return 0;
}
