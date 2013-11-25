#ifndef RL_METHOD_H
#define RL_METHOD_H

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
    PROBABILITY_MATCHING,
    EPS_GREEDY
};

class RLMethod
{
protected:
    ExperimentalModel *model;
    double eta; // critic learning rate
    double alpha; // actor learning rate
    double gamma; // discount factor 
    ActionSelectionMethod method; // action selection method
    double beta; // softmax temperature
    double min_R; // minimum reward for probability matching -- ??? #HACK #hack #FIXME
    double noise; // in what fraction of the cases will the monkey accidentally press the wrong button)
    double eps; // epsilon for epsilon-greedy action selection

    map<Choice*, double> policy;
    map<Choice*, double> H;
    map<State*, Choice*> optimal;

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
        Transition* result = NULL;
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
                result = trans;
                break;
            }
        }

        // noise -- press wrong button sometimes
        if (state->type == DETERMINISTIC)
        {
            double r = (double)rand() / RAND_MAX;
            if (r < noise)
            {
                int trans_idx = rand() % state->out.size();
                result = state->out[trans_idx];
            }
        }
        return result;
    }

    virtual Choice* GetOptimalChoice(State *state) = 0;

    virtual double GetChoiceWeight(Choice *choice) = 0;

    void UpdatePolicy(State *state)
    {
        if (state->type == PROBABILISTIC)
        {
            // no policy for non-choice transitions (i.e. non-deterministic states)
            return;
        }
        double total = 0;
        optimal[state] = GetOptimalChoice(state);
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

    // bookkeeping methods

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

    void Reset()
    {
        policy.clear();
        state_extras.clear();
        cue_extras.clear();
        transition_extras.clear();
        H.clear();
        optimal.clear();
        for (int i = 0; i < model->states.size(); i++)
        {
            State *state = model->states[i];
            optimal[state] = NULL;
            state_extras[state] = StateExtra();
            if (state->type == DETERMINISTIC)
            {
                double prob_avg = 1.0 / state->out.size();
                for (int j = 0; j < state->out.size(); j++)
                {
                    Choice *choice = dynamic_cast<Choice*>(state->out[j]);
                    policy[choice] = prob_avg;
                    optimal[state] = choice;
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

public:
    friend class Morris;

    RLMethod(ExperimentalModel *experiment_model,
        double critic_learning_rate,
        double actor_learning_rate,
        double discount_factor,
        ActionSelectionMethod action_selection_method,
        double softmax_temperature,
        double minimum_action_reward,
        double fraction_wrong_button,
        double epsilon_greedy_constant) :
        model(experiment_model),
        eta(critic_learning_rate),
        alpha(actor_learning_rate),
        gamma(discount_factor),
        method(action_selection_method),
        beta(softmax_temperature),
        min_R(minimum_action_reward),
        noise(fraction_wrong_button),
        eps(epsilon_greedy_constant)
    {
    }

    virtual void Trial(bool do_print) = 0;

    virtual void Print() = 0;

};





#endif
