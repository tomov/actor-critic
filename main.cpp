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

#include "model.h"

enum ActionSelectionMethod
{
    SOFTMAX,
    PROBABILITY_MATCHING
};

class ActorCritic
{
private:
    ExperimentalModel *model;
    double eta; // critic learning rate
    double alpha; // actor learning rate
    double gamma; // discount factor 
    ActionSelectionMethod method; // action selection method
    double beta; // softmax temperature
    double min_R; // minimum reward for probability matching -- ??? #HACK #hack #FIXME 

    map<Choice*, double> policy;
    map<State*, double> V;
    map<Choice*, double> H;

    struct StateExtra
    {
        int times; // how many times we passed that state
        StateExtra() : times(0) { }
    };
    map<State*, StateExtra> state_extras;

    struct TransitionExtra
    {
        double PE_avg; // average PE for transition
        int times;     // how many times the transition occured 
        double measured_probability;   // what is the real probability, measured in practice, of this transition happening vs. any of the other transitions from that origin state
        TransitionExtra() : PE_avg(0), times(0), measured_probability(0) { }
    };
    map<Transition*, TransitionExtra> transition_extras;

    struct CueExtra
    {
        double reward_avg; // average reward received for this cue
        int times;         // how many times we passed that cue
        double PE_avg;     // average PE for transitions going straight out of this cue
        CueExtra() : reward_avg(0), times(0), PE_avg(0) { }
    };
    map<Cue*, CueExtra> cue_extras;

    Transition* PickTransition(State *state)
    {
        double r = (double)rand() / RAND_MAX;
        double tot = 0;
        for (int i = 0; i < state->out.size(); i++)
        {
            Transition *trans = state->out[i];
            if (state->type == PROBABILISTIC)
            {
                Chance *chance = dynamic_cast<Chance*>(trans);
                tot += chance->probability;
            }
            else
            {
                Choice *choice = dynamic_cast<Choice*>(trans);
                tot += policy[choice];
            }
            if (tot >= r)
            {
                return trans;
            }
        }
        return NULL;
    }

    double GetChoiceWeight(Choice *choice)
    {
        switch(method)
        {
            case SOFTMAX:
            {
                return exp(beta * H[choice]);
            }
            break;
            case PROBABILITY_MATCHING:
            {
                State *S_new = choice->to;
                return max(V[S_new], min_R);
            }
            break;
            default:
            {
                return -1e100;
            }
        }
    }

    void UpdatePolicy(State *state)
    {
        if (state->type == PROBABILISTIC)
        {
            // no policy for non-choice transitions (i.e. non-deterministic states)
            return;
        }
        double total = 0;
        for (int i = 0; i < state->out.size(); i++)
        {
            Choice* choice = dynamic_cast<Choice*>(state->out[i]);
            double weight = GetChoiceWeight(choice);
            policy[choice] = weight;
            total += weight;
        }
        for (int i = 0; i < state->out.size(); i++)
        {
            Choice* choice = dynamic_cast<Choice*>(state->out[i]);
            policy[choice] /= total;
        }
    }

public:
    friend class Morris;

    void Reset()
    {
        policy.clear();
        V.clear();
        state_extras.clear();
        cue_extras.clear();
        transition_extras.clear();
        H.clear();
        for (int i = 0; i < model->states.size(); i++)
        {
            State *state = model->states[i];
            V[state] = 0;
            state_extras[state] = StateExtra();
            if (state->type == DETERMINISTIC)
            {
                double prob_avg = 1.0 / state->out.size();
                for (int j = 0; j < state->out.size(); j++)
                {
                    Choice *choice = dynamic_cast<Choice*>(state->out[j]);
                    policy[choice] = prob_avg;
                    H[choice] = 0;
                }
            }
        }
        for (int i = 0; i < model->transitions.size(); i++)
        {
            Transition *trans = model->transitions[i];
            transition_extras[trans] = TransitionExtra();
        }
        for (int i = 0; i < model->cues.size(); i++)
        {
            Cue *cue = model->cues[i];
            cue_extras[cue] = CueExtra();
        }
    }

    ActorCritic(ExperimentalModel *experiment_model,
        double critic_learning_rate,
        double actor_learning_rate,
        double discount_factor,
        ActionSelectionMethod action_selection_method,
        double softmax_temperature,
        double minimum_action_reward) :
        model(experiment_model),
        eta(critic_learning_rate),
        alpha(actor_learning_rate),
        gamma(discount_factor),
        method(action_selection_method),
        beta(softmax_temperature),
        min_R(minimum_action_reward)
    {
        Reset();
    }

    void UpdateAveragePE(Transition* trans, double PE)
    {
        transition_extras[trans].PE_avg = (transition_extras[trans].PE_avg * transition_extras[trans].times + PE) / (transition_extras[trans].times + 1);
        transition_extras[trans].times++;
        state_extras[trans->from].times++;
        transition_extras[trans].measured_probability = (double)transition_extras[trans].times / state_extras[trans->from].times;
    }

    void UpdateAverageReward(Cue* cue, double reward)
    {
        if (cue == NULL)
        {
            return;
        }
        cue_extras[cue].reward_avg = (cue_extras[cue].reward_avg * cue_extras[cue].times + reward) / (cue_extras[cue].times + 1);
        cue_extras[cue].times++;
    }

    void Trial(bool do_print)
    {
        if (do_print) cout<<"\n  ---------------------- TRIAL --------------\n\n";
        State* S = model->start;
        double PE_prev = 0;
        double PE_prev_prev = 0;
        map<Cue*, double> seen_cues;
        while (S != model->end)
        {
            // pick choice or chance and get new state
            Transition* a = PickTransition(S);
            State *S_new = a->to;
            double R_new = S_new->reward;

            // calculate prediciton error
            double PE = R_new + gamma * V[S_new] - V[S];

            // update state value
            V[S] += eta * PE;

            // update policy
            if (S->type == DETERMINISTIC)
            {
                H[dynamic_cast<Choice*>(a)] += alpha * PE;
            }
            UpdatePolicy(S);
            if (do_print) cout<<" from "<<S->name<<" to "<<S_new->name<<", PE = "<<PE<<"\n";

            // bookkeeping -- average PE per action & prob of chosing this action
            UpdateAveragePE(a, PE + PE_prev + PE_prev_prev);

            // bookkeeping -- average reward received per seen cue
            if (S->cue != NULL && seen_cues.find(S->cue) == seen_cues.end())
            {
                seen_cues[S->cue] = 0;
            }
            for (map<Cue*, double>::iterator it = seen_cues.begin(); it != seen_cues.end(); it++)
            {
                it->second += S->reward;
            }
           
            // move to new state
            PE_prev_prev = PE_prev;
            PE_prev = PE;
            S = S_new;
        }

        // update the average reward for all cues passed on this trial
        for (map<Cue*, double>::iterator it = seen_cues.begin(); it != seen_cues.end(); it++)
        {
            UpdateAverageReward(it->first, it->second);
        }
    }

    void Print()
    {
        cout<<"\n  States:\n";
        for (int i = 0; i < model->states.size(); i++)
        {
            State *state = model->states[i];
            cout<<"    V["<<state->name<<"] = "<<V[state]<<", times = "<<state_extras[state].times<<"\n";
        }
        cout<<"\n  Transitions:\n";
        for (int i = 0; i < model->transitions.size(); i++)
        {
            Transition *trans = model->transitions[i];
            cout<<"     "<<trans->from->name<<" -> "<<trans->to->name<<": ";
            if (trans->from->type == DETERMINISTIC)
            {
                Choice *choice = dynamic_cast<Choice*>(trans);
                cout<<"         ("<<choice->name<<")               policy = "<<policy[choice]<<", H = "<<H[choice];
            }
            cout<<", PE_avg = "<<transition_extras[trans].PE_avg<<", times = "<<transition_extras[trans].times<<", measured prob = "<<transition_extras[trans].measured_probability;
            cout<<"\n";
        }
        cout<<"\n  Cue\n";
        for (int i = 0; i < model->cues.size(); i++)
        {
            Cue* cue = model->cues[i];
            cout<<"    "<<cue->name<<": reward_avg = "<<cue_extras[cue].reward_avg<<", times = "<<cue_extras[cue].times<<"\n";
        }
        cout<<"\n";
    }

};


class Morris
{
private:
    ActorCritic *ac;
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

public:
    Morris(ActorCritic *actor_critic, double dopamine_bias) :
        ac(actor_critic),
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
            double obtained_reward = ac->cue_extras[cue].reward_avg;
            ss<<obtained_reward<<", "<<obtained_reward;
            y.push_back(ss.str());
        }
        PrintFigure<string, string>("2a", 2, 2, 1, "bar", x, y, "Reward probability", "Obtained reward (R) (%)", "legend('left', 'right');\n");
    }


    void Figure2b()
    {
        vector<double> x, y;
        // for each decision trial cue (e.g. 50-50, or 50-75, etc)
        // #hardcoded... FIXME
        for (int i = 4; i < 14; i++)
        {
            Cue *cue = ac->model->cues[i];
            // for each state for that cue (e.g. 75-50 and 50-75)
            for (int j = 0; j < cue->states.size(); j++)
            {
                State *state = cue->states[j];
                double R_right = GetAverageReferenceTrialRewardFromDecisionTrialAction(state->out[0]);
                double R_total = 0;
                for (int k = 0; k < state->out.size(); k++)
                {
                    R_total += GetAverageReferenceTrialRewardFromDecisionTrialAction(state->out[k]);
                }
                x.push_back(R_right / R_total);
                double C_right = ac->transition_extras[state->out[0]].measured_probability;
                y.push_back(C_right);
                x.push_back(1.0 - R_right / R_total);
                y.push_back(1.0 - C_right);
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
            double dopamine_response = GetAveragePE(cue);
            ostringstream ss;
            ss<<dopamine_response<<", "<<dopamine_response;
            y.push_back(ss.str());
        }
        PrintFigure<string, string>("2c", 2, 2, 2, "bar", x, y, "Reward probability", "PE ~ Dopamine response", "legend('left', 'right');\n");
    }


    void Figure2d()
    {
        vector<double> x, y;
        // for each decision trial cue (e.g. 50-50, or 50-75, etc)
        // #hardcoded... FIXME
        for (int i = 4; i < 14; i++)
        {
            Cue *cue = ac->model->cues[i];
            // for each state for that cue (e.g. 75-50 and 50-75)
            for (int j = 0; j < cue->states.size(); j++)
            {
                State *state = cue->states[j];
                double D_right = GetAverageReferenceTrialPEFromDecisionTrialAction(state->out[0]);
                double D_total = 0;
                for (int k = 0; k < state->out.size(); k++)
                {
                    D_total += GetAverageReferenceTrialPEFromDecisionTrialAction(state->out[k]);
                }
                x.push_back(D_right / D_total);
                double C_right = ac->transition_extras[state->out[0]].measured_probability;
                y.push_back(C_right);
                x.push_back(1.0 - D_right / D_total);
                y.push_back(1.0 - C_right);
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
        int cue_ids[] = {5, 7, 8, 9, 11, 12};
        for (int i = 0; i < 6; i++)
        {
            Cue *cue = ac->model->cues[cue_ids[i]];
            x.push_back("'" + cue->name + "'");
            ostringstream ss;
            // just one...
            for (int j = 0; j < cue->states.size(); j++)
            {
                State *state = cue->states[j];
                for (int k = (int)state->out.size() - 1; k >= 0; k--)
                {
                    Transition *trans = state->out[k];
                    double PE_avg = ac->transition_extras[trans].PE_avg;
                    ss<<PE_avg + bias<<", ";
                }
            }
            y.push_back(ss.str());
        }
        PrintFigure<string, string>("4b", 3, 2, 3, "bar", x, y, "State (pair)", "PE ~ Dopamine response", "legend('low', 'high');\n");
    }

};


int main()
{
    ExperimentalModel *model = new ExperimentalModel();

    model->Read();
    model->Print();

    ActorCritic *actor_critic = new ActorCritic(
        model, 
        /* eta = critic learning rate */ 0.5,
        /* alpha = actor learning rate */ 0.5,
        /* gamma = discount factor */ 0.99,
        /* action selection method */ PROBABILITY_MATCHING,
        /* beta = softmax temperature */ 0.01,
        /* min_R = minimum action reward */ 1);

    for (int i = 0; i < 30000; i++)
    {
        actor_critic->Trial(/* do_print */ false);
    }
    actor_critic->Print();

    Morris morris(actor_critic, /* dopamine/PE base line */ 75);
    morris.Figure2a();
    morris.Figure2b();
    morris.Figure2c();
    morris.Figure2d();

    cout<<"figure;\n";
    morris.Figure4a();
    morris.Figure4b();

    return 0;
}
