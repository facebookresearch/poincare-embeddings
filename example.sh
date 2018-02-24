#!/bin/sh

# Get number of threads from environment or set to default
if [ -z "$NTHREADS" ]; then
   NTHREADS=2
fi

echo "Using $NTHREADS threads"

# make sure OpenMP doesn't interfere with pytorch.multiprocessing
export OMP_NUM_THREADS=1

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
