#ifndef Q_LEARNING_H 
#define Q_LEARNING_H 

#include "sarsa.h"

class QLearning : public SARSA 
{
public:

    QLearning(ExperimentalModel *experiment_model,
        double critic_learning_rate,
        double actor_learning_rate,
        double discount_factor,
        ActionSelectionMethod action_selection_method,
        double softmax_temperature,
        double minimum_action_reward,
        double fraction_wrong_button,
        double epsilon_greedy_constant) :
        SARSA(experiment_model,
        critic_learning_rate,
        actor_learning_rate,
        discount_factor,
        action_selection_method,
        softmax_temperature,
        minimum_action_reward,
        fraction_wrong_button,
        epsilon_greedy_constant)
    { }

    void Trial(bool do_print)
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
            Transition *a_optimal = A_new;
            if (S_new->type == DETERMINISTIC)
            {
                a_optimal = GetOptimalChoice(S_new);
            }
            double PE = R_new + gamma * Q[a_optimal] - Q[A];
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

};


#endif
