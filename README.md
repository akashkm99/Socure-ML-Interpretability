# Socure-ML-Interpretability

The methods which were used in an attempt to explain the predictions of a deep model

- [ ] Gradients
- [ ] Integrated Gradients
- [ ] LIME
- [ ] SHAP
- [ ] Deep Lift
- [ ] [Causal Interpretation](https://arxiv.org/pdf/1902.02302.pdf)
- [ ] [Hierarchical Interpretability](https://openreview.net/pdf?id=SkEqro0ctQ)
- [ ] REINFORCE

# Execution Procedure

## Gradients & Integrated Gradients

`gradient_attributions.py` contains the code which uses Gradients and Integrated Gradient algorithms. To run the file
```
python gradient_attributions.py
```
## LIME

Implements the LIME interpretability technique

```
python lime_attributions.py
```

## SHAP & Deep Lift

Refer `captum.ipynb` where both these algorithms are used and graphs are also plotted for visual understanding

## REINFORCE

We use a reinforcement learning technique to select the features which are essential for the network and minimize the number of features selected with minimum reduction in accuracy (F-score & Binary cross entropy. We use policy gradients technique and predict a Bernoulli distribution for each feature (whether selected or rejected). The implementation can be found `RL_model.py`, `RL_train.py`.

## Causal Interpretation

This is a [recent paper](https://arxiv.org/pdf/1902.02302.pdf) which aims to address the causal influence of features on the predictions thereby giving an interpretation notion. The main theme of the paper lies along the lines of "Correlation doesn't imply Causation". The plots generated and the code can be seen in `ACE.ipynb`

## Hierarchical Interpretability

This is also another recent paper which gives a tree like interpretation to the features by effectively capturing the interaction and compositionality between the features. We used the implementation from [here](https://github.com/csinva/hierarchical-dnn-interpretations) to draw possible conclusions from the models.