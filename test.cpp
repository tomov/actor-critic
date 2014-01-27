#include "sarsa.h"

int main()
{
    // -------------------------------------------
    //                Read Experiment
    // -------------------------------------------

    ExperimentalModel *model = new ExperimentalModel();

    model->Read();
    model->Print();

    // -------------------------------------------
    //                Simulate Experiment
    // -------------------------------------------

    SARSA *sarsa = new SARSA(
        model, 
        /* eta = critic learning rate */ 0.1,
        /* alpha = actor learning rate */ 0.1,
        /* gamma = discount factor */ 0.99,
        /* action selection method */ PROBABILITY_MATCHING,
        /* beta = softmax temperature */ 0.01,
        /* min_R = minimum action reward */ 1,
        /* noise = fraction of wrong button presses */ 0.1,
        /* eps = epsilon constant for eps-greedy action selection */ 0.05);

    for (int i = 0; i < 30000; i++)
    {
        sarsa->Trial(/* do_print */ false);
    }
    sarsa->Print();



    return 0;
}
