# VILA attribution and this experiment

Sources checked: [paper v3](https://arxiv.org/html/2602.13670v3), author [models/vila.py](https://github.com/byzhaoAI/VILA/blob/ad4293af236d86e05d8e850d2ce33638fc16c7ed/models/vila.py), [utils/VILA.py](https://github.com/byzhaoAI/VILA/blob/ad4293af236d86e05d8e850d2ce33638fc16c7ed/utils/VILA.py).

The paper describes fixed Gaussian nonlinear features, dual visual fusion and candidate semantic enhancement. These are attributed to VILA. This experiment is an adaptation, not an exact reproduction.

| Item | Paper / author implementation | This fixed experiment |
|---|---|---|
| Adapted features | Paper 768-dimensional branch | APART main/few 1536 dimensions, normal two-class Task1 |
| Buffer | Paper 16384 | Fixed 4096, paired NumPy Gaussian blocks; no RF row renormalization |
| Objective | Ordinary sample ridge | Equal class objective, S + K lambda I |
| Selection | Paper LOOCV; code 80/20, grid 1e-8 through 1e8 | Task1 identity-component CV, nine fixed lambdas; already-adapted representation, not independent end-to-end validation |
| Semantics | Paper additive alpha=1; code 0.8s+0.2v | Both alpha=1 and alpha=.25 fixed; no winner selection |
| Initial W | Author code multiplies initial solution by .9 | No .9 factor |
| Encoders | Generic visual anchor | Generic and native BiomedCLIP kept separately |
| Update | Recursive analytic classification | Recursive sufficient statistics with task-end Cholesky; no inverse or neural optimization |
| Measure | Paper reports ordinary accuracy | Primary balanced accuracy; macro-F1 and ordinary accuracy separately |

Current scientific definitions are the supplied plan. Geometry controls, class balancing, projection seeds, medical templates, group CV and utility thresholds are this protocol's choices. Different encoder systems differ in corpus, tokenizer, architecture and preprocessing; effects cannot be attributed solely to medical pretraining. BiomedCLIP exposure remains UNKNOWN. Representation rigidity is a hypothesis, not an established unique mechanism in these data.
