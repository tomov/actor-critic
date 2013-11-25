#include "morris.h"

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

    ActorCritic *actor_critic = new ActorCritic(
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
        actor_critic->Trial(/* do_print */ false);
    }
    actor_critic->Print();

    // -------------------------------------------
    //                Print Results 
    // -------------------------------------------

    Morris morris(actor_critic, /* dopamine/PE base line */ 75);
    morris.Figure2a();
    morris.Figure2b();
    morris.Figure2c();
    morris.Figure2d();
    cout<<"figure;\n";
    morris.Figure4a();
    morris.Figure4b();
    morris.Figure4c();
    morris.Figure4d();
    morris.Figure4e();
    morris.Figure4f();

    return 0;
}
