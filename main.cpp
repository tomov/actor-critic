#include "morris.h"
#include "actor-critic.h"
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

    RLMethod *rl_method = new SARSA(
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
        rl_method->Trial(/* do_print */ false);
    }
    rl_method->Print();

    // -------------------------------------------
    //                Print Results 
    // -------------------------------------------

    Morris morris(rl_method, /* dopamine/PE base line */ 75);
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
