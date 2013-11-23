#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>
#include <numeric>
#include <functional>
#include <cmath>
#include <utility>

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

    struct TransitionExtra
    {
        double PE_avg;
        int times;
    };
    map<Transition*, TransitionExtra> transition_extras;

    struct CueExtra
    {
        double reward_avg;
        int times;
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
    void Reset()
    {
        policy.clear();
        V.clear();
        cue_extras.clear();
        transition_extras.clear();
        H.clear();
        for (int i = 0; i < model->states.size(); i++)
        {
            State *state = model->states[i];
            V[state] = 0;
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

    void Trial(bool do_print)
    {
        State* S = model->start;
        if (do_print) cout<<"\n  ---------------------- TRIAL --------------\n\n";
        while (S != model->end)
        {
            Transition* a = PickTransition(S);
            State *S_new = a->to;
            double R_new = S_new->reward;

            double PE = R_new + gamma * V[S_new] - V[S];
            V[S] += eta * PE;

            if (S->type == DETERMINISTIC)
            {
                H[dynamic_cast<Choice*>(a)] += alpha * PE;
            }
            UpdatePolicy(S);
            if (do_print) cout<<" from "<<S->name<<" to "<<S_new->name<<", PE = "<<PE<<"\n";

            S = S_new;
        }
    }

    void Print()
    {
        cout<<"\n  States:\n";
        for (int i = 0; i < model->states.size(); i++)
        {
            State *state = model->states[i];
            cout<<"    V["<<state->name<<"] = "<<V[state]<<"\n";
        }
        cout<<"\n  Choices:\n";
        for (int i = 0; i < model->transitions.size(); i++)
        {
            Transition *trans = model->transitions[i];
            if (trans->from->type == DETERMINISTIC)
            {
                cout<<"     "<<trans->from->name<<" -> "<<trans->to->name<<": ";
                Choice *choice = dynamic_cast<Choice*>(trans);
                cout<<"                policy = "<<policy[choice]<<", H = "<<H[choice];
                cout<<"\n";
            }
        }
        cout<<"\n";
    }

};

int main()
{
    ExperimentalModel *model = new ExperimentalModel();

    model->Read();
    model->Print();

    ActorCritic actor_critic(model, 
        /* eta = critic learning rate */ 0.01,
        /* alpha = actor learning rate */ 0.01,
        /* gamma = discount factor */ 0.99,
        PROBABILITY_MATCHING,
        /* beta = softmax temperature */ 0.01,
        /* min_R = minimum action reward */ 1);

    for (int i = 0; i < 10000; i++)
    {
        actor_critic.Trial(/* do_print */ false);
    }
    actor_critic.Print();

    return 0;
}
