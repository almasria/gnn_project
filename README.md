## Investigating the Transfer Learning capability of PINNsFormers


### Description
This repository is a fork from the PINNsFormer repository and holds additional 
experiments on transfer learning conducted for a project on Generative Neural 
Networks for the Sciences class at Heidelberg University.


### Code Structure 
To conduct transfer learning experiments, we adjusted the given models to include
additional parameters of the differential equations we were modeling. The corresponding 
implementation can be found in the `model_parametrized` directory.

The `transfer_learning` directory holds notebooks for the conducted transfer learning experiments and has the following directories:
- `1d_logistic_ode` - all experiments (pinn and pinnsformer on low and representative training data ranges)
- `1d_reaction` - all experiments (pinn and pinnsformer on low and representative training data ranges)
- `convection` - some experiments
- `demo_training_loops` - code snippets/notebooks used for prototyping/experimentation and code demonstration from the initial stages of the project.

The rest of the repo contains the implementation provided by the authors of the original paper.