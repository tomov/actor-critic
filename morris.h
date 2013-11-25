#ifndef MORRIS_H
#define MORRIS_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>
#include <numeric>
#include <functional>
#include <cmath>
#include <utility>
#include <cassert>

#include "rl-method.h"

class Morris
{
private:
    RLMethod *ac;
    double bias;
    
    template<typename X, typename Y>
    void PrintFigure(
        string name,
        int subplot_m,
        int subplot_n,
        int subplot_p,
        string plot_fn,
        vector<X> x,
        vector<Y> y,
        string xlabel,
        string ylabel,
        string extra_comands = "")
   {
        cout<<"\n %% ------ Figure "<<name<<" ------\n\n";
        char open_par = '[';
        char close_par = ']';
        if (plot_fn == "bar")
        {
            open_par = '{';
            close_par = '}';
        }
        cout<<"x_"<<name<<" = "<<open_par;
        for (int i = 0; i < x.size(); i++)
        {
            cout<<x[i]<<"; ";
        }
        cout<<close_par<<";\n";
        cout<<"y_"<<name<<" = [";
        for (int i = 0; i < y.size(); i++)
        {
            cout<<y[i]<<"; ";
        }
        cout<<"];\n";
        cout<<"subplot("<<subplot_m<<","<<subplot_n<<","<<subplot_p<<");\n";
        if (plot_fn == "bar")
        {
            cout<<plot_fn<<"(y_"<<name<<");\n";
            cout<<"set(gca, 'XTickLabel', x_"<<name<<");\n";
        }
        else
        {
            cout<<plot_fn<<"(x_"<<name<<", y_"<<name<<");\n";
        }
        cout<<"xlabel('"<<xlabel<<"');\n";
        cout<<"ylabel('"<<ylabel<<"');\n";
        cout<<extra_comands<<"\n";
        cout<<"\n";
    }

    double GetAverageReferenceTrialRewardFromDecisionTrialAction(Transition* trans)
    {
        // get the corresponding reference trial cue
        // from the type of reward that this action leads to
        // FIXME this is a #HACK -- we just store the queue in the extra
        // string of the reward state... super awk but that's the least
        // annoying way I could come up with
        Cue *ref_cue = ac->model->cue_from_name[trans->to->extra];
        return ac->cue_extras[ref_cue].reward_avg;
    }

    double GetAveragePE(State *state)
    {
        double PE_avg = 0;
        int times_total = 0;
        for (int j = 0; j < state->out.size(); j++)
        {
            Transition *trans = state->out[j];
            PE_avg += ac->transition_extras[trans].PE_avg * ac->transition_extras[trans].times;
            times_total += ac->transition_extras[trans].times;
        }
        assert(times_total == ac->state_extras[state].times);
        PE_avg /= ac->state_extras[state].times;
        return PE_avg + bias;
    }

    double GetAveragePE(Cue *cue)
    {
        double PE_avg = 0;
        int times_total = 0;
        for (int i = 0; i < cue->states.size(); i++)
        {
            State *state = cue->states[i];
            PE_avg += GetAveragePE(state) * ac->state_extras[state].times;
            times_total += ac->state_extras[state].times;
        }
        assert(times_total == ac->cue_extras[cue].times);
        PE_avg /= ac->cue_extras[cue].times;
        return PE_avg;
    }

    double GetAverageReferenceTrialPEFromDecisionTrialAction(Transition *trans)
    {
        // get the corresponding reference trial cue
        // from the type of reward that this action leads to
        // FIXME this is a #HACK -- we just store the queue in the extra
        // string of the reward state... super awk but that's the least
        // annoying way I could come up with
        Cue *ref_cue = ac->model->cue_from_name[trans->to->extra];
        return GetAveragePE(ref_cue);
    }

    double GetAveragePEForRewardedTransitionsFrom(State *state)
    {
        double PE_avg = 0;
        int times = 0;
        // FIXME this is a hack -- the reward is delayed (i.e. there are intermediate states,
        // like reward-25 --> wait --> wait --> wait --> reward-25-real --> juice or no-juice
        // we keep going until we hit the juice
        while (state->out.size() == 1)
        {
            state = state->out[0]->to;
        }
        for (int i = 0; i < state->out.size(); i++)
        {
            Transition* trans = state->out[i];
            if (trans->to->reward > 0)
            {
                PE_avg += ac->transition_extras[trans].PE_avg * ac->transition_extras[trans].times;
                times += ac->transition_extras[trans].times;
            }
        }
        if (times == 0)
        {
            return bias;
        }
        PE_avg /= times;
        return PE_avg + bias;
    }

    double GetAveragePEForRewardedTransitionsFromChildrenOf(State *state)
    {
        double PE_avg = 0;
        int times = 0;
        // for each action to a reward state (e.g. reward-25)
        for (int k = 0; k < state->out.size(); k++)
        {
            Transition* trans = state->out[k];
            // add the PE for actual reward delivery from that reward state
            PE_avg += GetAveragePEForRewardedTransitionsFrom(trans->to) * ac->transition_extras[trans].times;
            times += ac->transition_extras[trans].times;
        }
        assert(times == ac->state_extras[state].times);
        if (times == 0)
        {
            return 0;
        }
        PE_avg /= times;
        return PE_avg;
    }

    double GetAveragePEForRewardedTransitionsFromChildrenOf(Cue *cue)
    {
        double PE_avg;
        int times = 0;
        for (int j = 0; j < cue->states.size(); j++)
        {
            State *state = cue->states[j];
            PE_avg += GetAveragePEForRewardedTransitionsFromChildrenOf(state) * ac->state_extras[state].times;
            times += ac->state_extras[state].times;
        }
        assert(times == ac->cue_extras[cue].times);
        PE_avg /= times;
        return PE_avg;
    }



public:
    Morris(RLMethod *rl_method, double dopamine_bias) :
        ac(rl_method),
        bias(dopamine_bias)
    { }

    void Figure2a()
    {
        vector<string> x;
        vector<string> y;
        for (int i = 0; i < 4; i++)
        {
            Cue *cue = ac->model->cues[i];
            x.push_back("'" + cue->name + "'");
            ostringstream ss;
            for (int j = 0; j < cue->states.size(); j++)
            {
                State *state = cue->states[j];
                double obtained_reward = ac->state_extras[state].reward_avg;
                ss<<obtained_reward<<", ";
            }
            y.push_back(ss.str());
        }
        PrintFigure<string, string>("2a", 2, 2, 1, "bar", x, y, "Reward probability", "Obtained reward (R) (%)", "legend('left', 'right');\n");
    }


    void Figure2b()
    {
        vector<double> x, y;
        // #hardcoded FIXME which transition is right
        int right_action_idx = 1;
        // for each decision trial cue (e.g. 50-50, or 50-75, etc)
        // #hardcoded... FIXME
        for (int i = 4; i < 14; i++)
        {
            Cue *cue = ac->model->cues[i];
            // for each state for that cue (e.g. 75-50 and 50-75)
            for (int j = 0; j < cue->states.size(); j++)
            {
                State *state = cue->states[j];
                double R_right = GetAverageReferenceTrialRewardFromDecisionTrialAction(state->out[right_action_idx]);
                double R_total = 0;
                for (int k = 0; k < state->out.size(); k++)
                {
                    R_total += GetAverageReferenceTrialRewardFromDecisionTrialAction(state->out[k]);
                }
                x.push_back(R_right / R_total);
                double C_right = ac->transition_extras[state->out[right_action_idx]].measured_probability;
                y.push_back(C_right);
            }
        }
        PrintFigure<double, double>("2b", 2, 2, 3, "scatter", x, y, "R_{right} / (R_{right} + R_{left})", "C_{right}", "axis([0 1 0 1]);\nlsline;\n");
    }


    void Figure2c()
    {
        vector<string> x;
        vector<string> y;
        for (int i = 0; i < 4; i++)
        {
            Cue *cue = ac->model->cues[i];
            x.push_back("'" + cue->name + "'");
            ostringstream ss;
            for (int j = 0; j < cue->states.size(); j++)
            {
                State* state = cue->states[j];
                double dopamine_response = GetAveragePE(state);
                ss<<dopamine_response<<", ";
            }
            y.push_back(ss.str());
        }
        PrintFigure<string, string>("2c", 2, 2, 2, "bar", x, y, "Reward probability", "PE ~ Dopamine response", "legend('left', 'right');\n");
    }


    void Figure2d()
    {
        vector<double> x, y;
        // #hardcoded FIXME which transition is right
        int right_action_idx = 1;
        // for each decision trial cue (e.g. 50-50, or 50-75, etc)
        // #hardcoded... FIXME
        for (int i = 4; i < 14; i++)
        {
            Cue *cue = ac->model->cues[i];
            // for each state for that cue (e.g. 75-50 and 50-75)
            for (int j = 0; j < cue->states.size(); j++)
            {
                State *state = cue->states[j];
                double D_right = GetAverageReferenceTrialPEFromDecisionTrialAction(state->out[right_action_idx]);
                double D_total = 0;
                for (int k = 0; k < state->out.size(); k++)
                {
                    D_total += GetAverageReferenceTrialPEFromDecisionTrialAction(state->out[k]);
                }
                x.push_back(D_right / D_total);
                double C_right = ac->transition_extras[state->out[right_action_idx]].measured_probability;
                y.push_back(C_right);
            }
        }
        PrintFigure<double, double>("2d", 2, 2, 4, "scatter", x, y, "D_{right} / (D_{right} + D_{left})", "C_{right}", "axis([0 1 0 1]);\nlsline;\n");
    }

    void Figure4a()
    {
        vector<string> x;
        vector<double> y;
        // #hardcoded FIXME
        for (int i = 4; i < 14; i++)
        {
            Cue *cue = ac->model->cues[i];
            x.push_back("'" + cue->name + "'");
            y.push_back(GetAveragePE(cue));
        }
        PrintFigure<string, double>("4a", 3, 2, 1, "bar", x, y, "State (pair)", "PE ~ Dopamine response");
    }

    void Figure4b()
    {
        vector<string> x;
        vector<string> y;
        // #hardcoded FIXME
        int left_action_idx = 0;
        int right_action_idx = 1;
        // #hardcoded FIXME
        int cue_ids[] = {5, 7, 8, 9, 11, 12};
        // for each decision cue with distinct outcomes
        for (int i = 0; i < 6; i++)
        {
            Cue *cue = ac->model->cues[cue_ids[i]];
            x.push_back("'" + cue->name + "'");
            double high_PE_avg = 0;
            double low_PE_avg = 0;
            for (int j = 0; j < cue->states.size(); j++)
            {
                State *state = cue->states[j];
                // #hardcoded FIXME
                Transition *trans_left = state->out[left_action_idx];
                Transition *trans_right = state->out[right_action_idx];
                Cue *cue_left = ac->model->cue_from_name[trans_left->to->extra];
                Cue *cue_right = ac->model->cue_from_name[trans_right->to->extra];
                if (cue_left->value > cue_right->value)
                {
                    high_PE_avg += ac->transition_extras[trans_left].PE_avg;
                    low_PE_avg += ac->transition_extras[trans_right].PE_avg;
                }
                else
                {
                    high_PE_avg += ac->transition_extras[trans_right].PE_avg;
                    low_PE_avg += ac->transition_extras[trans_left].PE_avg;
                }
            }
            high_PE_avg /= cue->states.size();
            low_PE_avg /= cue->states.size();
            ostringstream ss;
            ss<<high_PE_avg + bias<<", "<<low_PE_avg + bias;
            y.push_back(ss.str());
        }
        PrintFigure<string, string>("4b", 3, 2, 3, "bar", x, y, "State (pair)", "PE ~ Dopamine response", "legend('high', 'low');\n");
    }

    void Figure4c()
    {
        vector<double> x, y;
        // reference trials
        for (int i = 0; i < 4; i++)
        {
            Cue *cue = ac->model->cues[i];
            x.push_back(ac->cue_extras[cue].reward_avg);
            y.push_back(GetAveragePE(cue));
        }

        // decision trials
        for (int i = 0; i < 4; i++)
        {
            Cue *cue = ac->model->cues[i];
            double PE_avg = 0;
            int total = 0;
            for (int j = 0; j < ac->model->transitions.size(); j++)
            {
                Transition* trans = ac->model->transitions[j];
                // if it's an action in a decision trial
                if (ac->model->cue_from_name.find(trans->to->extra) != ac->model->cue_from_name.end())
                {
                    Cue* ref_cue = ac->model->cue_from_name[trans->to->extra];
                    // that corresponds to the same reference cue
                    if (ref_cue == cue)
                    {
                        PE_avg += ac->transition_extras[trans].PE_avg * ac->transition_extras[trans].times;
                        total += ac->transition_extras[trans].times;
                    }
                }
            }
            PE_avg /= total;
            x.push_back(cue->value);
            y.push_back(PE_avg + bias);
        }

        PrintFigure<double, double>("4c", 3, 2, 5, "scatter", x, y, "Action value", "PE ~ Dopamine response", "lsline;\nhold on;\nscatter(x_4c(5:end), y_4c(5:end), 'fill', 'blue');\nhold off;\n");
    }

    void Figure4d()
    {
        vector<string> x;
        vector<double> y;
        // for each decision cue
        for (int i = 4; i < 14; i++)
        {
            Cue* cue = ac->model->cues[i];
            x.push_back("'" + cue->name + "'");
            y.push_back(GetAveragePEForRewardedTransitionsFromChildrenOf(cue));
        }
        PrintFigure<string, double>("4d", 3, 2, 2, "bar", x, y, "State (pair)", "PE ~ Dopamine response");
    }

    void Figure4e()
    {
        vector<string> x;
        vector<string> y;
        // #hardcoded FIXME
        int left_action_idx = 0;
        int right_action_idx = 1;
        // for each decision cue
        // #hardcoded FIXME
        int cue_ids[] = {5, 7, 8, 9, 11, 12};
        for (int i = 0; i < 6; i++)
        {
            Cue *cue = ac->model->cues[cue_ids[i]];
            double high_PE_avg = 0;
            double low_PE_avg = 0;
            x.push_back("'" + cue->name + "'");
            // for each state
            for (int j = 0; j < cue->states.size(); j++)
            {
                State *state = cue->states[j];
                // #hardcoded FIXME
                Transition *trans_left = state->out[left_action_idx];
                Transition *trans_right = state->out[right_action_idx];
                Cue *cue_left = ac->model->cue_from_name[trans_left->to->extra];
                Cue *cue_right = ac->model->cue_from_name[trans_right->to->extra];
                if (cue_left->value > cue_right->value)
                {
                    high_PE_avg += GetAveragePEForRewardedTransitionsFrom(trans_left->to);
                    low_PE_avg += GetAveragePEForRewardedTransitionsFrom(trans_right->to);
                }
                else
                {
                    high_PE_avg += GetAveragePEForRewardedTransitionsFrom(trans_right->to); 
                    low_PE_avg += GetAveragePEForRewardedTransitionsFrom(trans_left->to);
                }
            }
            high_PE_avg /= cue->states.size();
            low_PE_avg /= cue->states.size();
            ostringstream ss;
            ss<<high_PE_avg<<", "<<low_PE_avg;
            y.push_back(ss.str());
        }
        PrintFigure<string, string>("4e", 3, 2, 4, "bar", x, y, "State (pair)", "PE ~ Dopamine response", "legend('high', 'low');\n");
    }

    void Figure4f()
    {
        vector<double> x, y;
        // reference trials
        for (int i = 0; i < 4; i++)
        {
            Cue *cue = ac->model->cues[i];
            x.push_back(ac->cue_extras[cue].reward_avg);
            y.push_back(GetAveragePEForRewardedTransitionsFromChildrenOf(cue));
        }

        // decision trials
        for (int i = 0; i < 4; i++)
        {
            Cue *cue = ac->model->cues[i];
            double PE_avg = 0;
            int total = 0;
            for (int j = 0; j < ac->model->transitions.size(); j++)
            {
                Transition* trans = ac->model->transitions[j];
                // if it's an action in a decision trial (i.e. leads to a reward state)
                if (ac->model->cue_from_name.find(trans->to->extra) != ac->model->cue_from_name.end())
                {
                    Cue* ref_cue = ac->model->cue_from_name[trans->to->extra];
                    // that leads corresponds to the same reference cue
                    if (ref_cue == cue)
                    {
                        // add the PE for actual reward delivery from that reward state  
                        PE_avg += GetAveragePEForRewardedTransitionsFrom(trans->to) * ac->transition_extras[trans].times;
                        total += ac->transition_extras[trans].times;
                    }
                }
            }
            PE_avg /= total;
            x.push_back(cue->value);
            y.push_back(PE_avg);
        }
        PrintFigure<double, double>("4f", 3, 2, 6, "scatter", x, y, "Action value", "PE ~ Dopamine response", "lsline;\nhold on;\nscatter(x_4f(5:end), y_4f(5:end), 'fill', 'blue');\nhold off;\n");
    }

};



#endif
