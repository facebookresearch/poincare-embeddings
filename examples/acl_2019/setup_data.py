#!/usr/bin/env python3

from tempfile import NamedTemporaryFile
from subprocess import check_call
import pandas
import os
import numpy as np


if not os.path.exists('hearst_ppmi_filtered.csv'):
    # Download Hearst counts data
    with NamedTemporaryFile(suffix='.txt.gz') as tfile:
        url = 'https://github.com/facebookresearch/hypernymysuite/raw/master/hearst_counts.txt.gz'  # noqa B950
        check_call(['wget', url, '-O', tfile.name])
        df = pandas.read_csv(tfile.name, header=None, sep='\t')
        df.columns = ['id1', 'id2', 'count']

    pw = df.rename(columns={'count': 'pw'}).groupby('id1')['pw'].sum()
    df = df.merge(pw.to_frame(), on=['id1'], how='left')
    pc = df.rename(columns={'count': 'pc'}).groupby('id2')['pc'].sum()
    df = df.merge(pc.to_frame(), on=['id2'], how='left')
    N = len(np.unique(df[['id1', 'id2']].values.reshape(-1)))
    df['ppmi'] = np.clip(
        np.log(N) + np.log(df['count']) - np.log(df['pw']) - np.log(df['pc']),
        0,
        1e12
    )
    df = df.rename(columns={'ppmi': 'weight'})
    df[df['weight'] > 0][['id1', 'id2', 'weight']]\
        .to_csv('hearst_ppmi_filtered.csv', index=False)

if not os.path.exists('data/bless.tsv'):
    data_dir = os.environ.get('HYPERNYMY_DATA_DIR', 'data')
    if not os.path.exists(os.path.join('data', 'bless.tsv')):
        print('Downloading hypernymysuite eval data...')
        url = 'https://raw.githubusercontent.com/facebookresearch/hypernymysuite/master/download_data.sh'  # noqa B950
        env = {**os.environ, 'HYPERNYMY_DATA_OUTPUT': 'data'}
        res = check_call(f'wget -q -O - {url} | bash', shell=True, env=env)
        if res != 0:
            raise ValueError('Failed to process validation/test data!')
