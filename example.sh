#!/bin/sh

# Get number of threads from environment or set to default
if [ -z "$NTHREADS" ]; then
   NTHREADS=2
fi

echo "Using $NTHREADS threads"

python3 embed.py \
       -dim 5 \
       -lr 0.3 \
       -epochs 200 \
       -negs 50 \
       -burnin 10 \
       -nproc "${NTHREADS}" \
       -distfn poincare \
       -dset wordnet/mammal_closure.tsv \
       -fout mammals.pth \
       -batchsize 10 \
       -eval_each 1 \
