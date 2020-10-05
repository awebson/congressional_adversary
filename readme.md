This repository accompanies our EMNLP 2020 paper “Are ‘Undocumented Workers’ the Same as ‘Illegal Aliens’? Probing and Disentangling Denotation and Connotation in Vector Spaces”. 

# Setup
(Python 3.7 or higher required.)
```
git clone https://github.com/awebson/congressional_adversary
cd congressional_adversary
python3 -m venv congressional_env  # or your preferred virtual environment solution
source congressional_env/bin/activate
pip install -r requirements.txt
pip install -e . 
mkdir data
```
Then, download the training and evaluation data from this [Google Drive link](https://drive.google.com/file/d/1D_YuVIRSbcfWlQrWDN1CSbo_bfx_x4YQ/view?usp=sharing), extract with your favorite tar command, and copy them to the `data` directory you just made. 

Note that this data is already fully preprocessed, so you don’t need to actually run any preprocessing script included in this repository. If you do, the raw corpus of Congressional Record is available from [Gentzkow et al. (2019)](https://data.stanford.edu/congress_text). The raw corpus of Partisan News is available from [Kiesel et al. (2019)](http://alt.qcri.org/semeval2019/index.php?id=tasks). 

# Training
Run `src/models/ideal_grounding.py` for the CR bill and CR topic models, or `src/models/proxy_grounding.py` for the CR proxy and PN proxy models. (See paper Sections 3 and 4 for more details.) Each model source file contains a config `dataclass` where you can change the default parameters as well as the command line arguments. Hyperparameters with reproducible results are documented in paper Appendix A. Run `tensorboard --logdir .` to see the results of your experiments.

## Why is this repository codenamed “Congressional Adversary”?
In typical lame academic humor, I thought it's funny that I implemented an adversarial neural net for Members of Congress, who are often adversarial, if not acrimonious, to each other. 

We were also deciding between “Adversarial Congress” and “Congressional Adversary”. The former sounds like some political science book that refutes the deliberative theory of democracy, whereas the latter just sounds like someone’s evil archenemy. Since this was a paper submitted to Empirical Methods in Natural Language Processing, not the American Journal of Political Science, we went with “Congressional Adversary”.
