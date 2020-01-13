# Inferring Concept Hierarchies from Text Corpora via Hyperbolic Embeddings

This document provides instructions for reproducing our results in ["Inferring Concept Hierarchies from Text Corpora via Hyperbolic Embeddings"](https://arxiv.org/abs/1902.00913)


### Get Data

Download NLTK's Wordnet corpous

```
python -c "import nltk; nltk.download('wordnet')"
```

Fetch Hearst PPMI data and validation/test data for hypernymy detection tasks

```
python examples/acl_2019/setup_data.py
```

### Train the model

```
python embed.py \
    -checkpoint chkpnt.bin \
    -dset hearst_ppmi_filtered.csv \
    -dim 16 \
    -manifold lorentz \
    -model entailment_cones \
    -lr 48.21816484591745 \
    -epochs 700 \
    -batchsize 281 \
    -negs 195 \
    -burnin 275 \
    -dampening 0.75 \
    -ndproc 8 \
    -eval_each 1 \
    -gpu 0 \
    -maxnorm 500000 \
    -burnin_multiplier 1.0 \
    -neg_multiplier 1.0 \
    -lr_type constant \
    -train_threads 1 \
    -margin 0.6152963538446055 \
    -eval hypernymy
```

### Evaluate the model

```
python hypernymy_eval.py chkpnt.bin.best
```

### Using our pre-trained model.

Alternatively, you can download a pre-trained model [here](dl.fbaipublicfiles.com/hype/acl_2019_chkpnt.bin.best)

```
python hypernymy_eval.py acl_2019_chkpnt.bin.best 
```