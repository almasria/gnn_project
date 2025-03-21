## Investigating the Transfer Learning capability of PINNsFormers


### Description
This repository is a fork from the PINNsFormer repository and holds additional 
experiments on transfer learning conducted for a project on Generative Neural 
Networks for the Sciences class at Heidelberg University.


### Code Structure 
To conduct transfer learning experiments, we adjusted the given models to include
additional parameters of the differential equations we were modeling. The corresponding 
implementation can be found in the `model_parametrized` directory.

The `transfer_learning` directory holds the conducted evaluation notebooks:
- `1d_logistic_ode` 
- `1d_reaction`
- `convection`
- `demo_training_loops`
- `mixed`

The rest of the repo contains the implementation provided by the authors of the original paper.