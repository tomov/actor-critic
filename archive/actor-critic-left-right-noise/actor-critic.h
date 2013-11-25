#ifndef ACTOR_CRITIC_H
#define ACTOR_CRITIC_H

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
    double noise; // in what fraction of the cases will the monkey accidentally press the wrong button)

    map<Choice*, double> policy;
    map<State*, double> V;
    map<Choice*, double> H;

    struct StateExtra
    {
        int times; // how many times we passed that state
        double reward_avg; // average reward received for this state ONLY IF it is a cue state
        int reward_times; // how many times we passed that state; for cue states only; used for calculating reward_avg
        StateExtra() : times(0), reward_avg(0), reward_times(0) { }
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

    void UpdateAverageReward(State *state, double reward)
    {
        if (state == NULL)
        {
            return;
        }
        state_extras[state].reward_avg = (state_extras[state].reward_avg * state_extras[state].reward_times + reward) / (state_extras[state].reward_times + 1);
        state_extras[state].reward_times++;
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
        double minimum_action_reward,
        double fraction_wrong_button) :
        model(experiment_model),
        eta(critic_learning_rate),
        alpha(actor_learning_rate),
        gamma(discount_factor),
        method(action_selection_method),
        beta(softmax_temperature),
        min_R(minimum_action_reward),
        noise(fraction_wrong_button)
    {
        Reset();
    }

    void Trial(bool do_print)
    {
        if (do_print) cout<<"\n  ---------------------- TRIAL --------------\n\n";
        State* S = model->start;
        double PE_prev = 0;
        double PE_prev_prev = 0;
        map<Cue*, double> seen_cues;
        map<State*, double> seen_cue_states;
        while (S != model->end)
        {
            // pick choice or chance and get new state
            Transition* a = PickTransition(S);

            // noise
            if (S->type == DETERMINISTIC)
            {
                double r = (double)rand() / RAND_MAX;
                if (r < noise)
                {
                    int a_idx = rand() % S->out.size();
                    a = S->out[a_idx];
                }
            }

            // calculate prediciton error
            State *S_new = a->to;
            double R_new = S_new->reward;
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
            //PE_prev_prev = PE_prev;
            PE_prev = PE;
            S = S_new;
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
            cout<<"    V["<<state->name<<"] = "<<V[state]<<", times = "<<state_extras[state].times<<", reward_avg = "<<state_extras[state].reward_avg<<", reward times = "<<state_extras[state].reward_times<<"\n";
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


#endif
