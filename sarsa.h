#ifndef SARSA_H 
#define SARSA_H 

#include "rl-method.h"

class SARSA : public RLMethod
{
protected:
    map<Transition*, double> Q;

    Choice* GetOptimalChoice(State *state)
    {
        if (state->type == PROBABILISTIC)
        {
            return NULL;
        }
        Transition* result = NULL;
        double Q_max = -1e100;
        for (int i = 0; i < state->out.size(); i++)
        {
            Transition *trans = state->out[i];
            if (Q[trans] > Q_max)
            {
                result = trans;
                Q_max = Q[trans];
            }
        }
        return dynamic_cast<Choice*>(result);
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
                return max(Q[choice], min_R);
            }
            break;
            case EPS_GREEDY:
            {
                if (choice == optimal[choice->from])
                {
                    return 1 - eps;
                }
                return eps / choice->from->out.size();
            }
            default:
            {
                return -1e100;
            }
        }
    }

public:

    void Reset()
    {
        RLMethod::Reset();
        Q.clear();
        for (int i = 0; i < model->transitions.size(); i++)
        {
            Transition *trans = model->transitions[i];
            Q[trans] = 0;
        }
    }

    SARSA(ExperimentalModel *experiment_model,
        double critic_learning_rate,
        double actor_learning_rate,
        double discount_factor,
        ActionSelectionMethod action_selection_method,
        double softmax_temperature,
        double minimum_action_reward,
        double fraction_wrong_button,
        double epsilon_greedy_constant) :
        RLMethod(experiment_model,
        critic_learning_rate,
        actor_learning_rate,
        discount_factor,
        action_selection_method,
        softmax_temperature,
        minimum_action_reward,
        fraction_wrong_button,
        epsilon_greedy_constant)
    { 
        // must call its own Reset() method
        // we cannot call it from the base class b/c in the constructor,
        // the class is not yet created, so it cannot call the derived method
        // see here --> http://stackoverflow.com/questions/4073210/calling-an-overridden-function-from-a-base-class
        Reset();
    }

    virtual void Trial(bool do_print)
    {
        if (do_print) cout<<"\n  ---------------------- TRIAL --------------\n\n";
        State *S = model->start;
        Transition *A = PickTransition(S);

        double PE_prev = 0;
        map<Cue*, double> seen_cues;
        map<State*, double> seen_cue_states;
        while (S != model->end)
        {
            State *S_new = A->to;
            Transition *A_new = PickTransition(S_new);

            double R_new = S_new->reward;
            double PE = R_new + gamma * Q[A_new] - Q[A];
            Q[A] += eta * PE;

            // update policy
            if (S->type == DETERMINISTIC)
            {
                H[dynamic_cast<Choice*>(A)] += alpha * PE;
            }
            UpdatePolicy(S);
            if (do_print) cout<<" from "<<S->name<<" to "<<S_new->name<<", PE = "<<PE<<"\n";

            // bookkeeping -- average PE per action & prob of chosing this action
            UpdateAveragePE(A, PE + PE_prev);

            // bookkeeping -- average reward received per seen cue
            if (S->cue != NULL && seen_cues.find(S->cue) == seen_cues.end())
            {
                seen_cues[S->cue] = 0;
            }
            for (map<Cue*, double>::iterator it = seen_cues.begin(); it != seen_cues.end(); it++)
            {
                it->second += S->reward;
            }

            // bookkeeping -- average reward received per seen cue state (similar to above)
            if (S->cue != NULL && seen_cue_states.find(S) == seen_cue_states.end())
            {
                seen_cue_states[S] = 0;
            }
            for (map<State*, double>::iterator it = seen_cue_states.begin(); it != seen_cue_states.end(); it++)
            {
                it->second += S->reward;
            }
          
            // move to new state
            PE_prev = PE;
            S = S_new;
            A = A_new;
        }
        // bookkeeping -- update the average reward for all cues passed on this trial
        for (map<Cue*, double>::iterator it = seen_cues.begin(); it != seen_cues.end(); it++)
        {
            UpdateAverageReward(it->first, it->second);
        }
        // bookkeeping -- same for cue states
        for (map<State*, double>::iterator it = seen_cue_states.begin(); it != seen_cue_states.end(); it++)
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
            cout<<"    optimal["<<state->name<<"] = "<<(optimal[state] ? optimal[state]->name : "None")<<", times = "<<state_extras[state].times<<", reward_avg = "<<state_extras[state].reward_avg<<", reward times = "<<state_extras[state].reward_times<<"\n";
        }
        cout<<"\n  Transitions:\n";
        for (int i = 0; i < model->transitions.size(); i++)
        {
            Transition *trans = model->transitions[i];
            cout<<"     Q["<<trans->from->name<<" -> "<<trans->to->name<<"] = "<<Q[trans]<<": ";
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


#endif
