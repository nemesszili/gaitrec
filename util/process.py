import pandas as pd

import multiprocessing as mp
import time
import os
import csv
from itertools import islice

from util.const import LINES_CHUNK, DATASET_PATH, g, COLS, NORM_COLS
import util.features as ft

def export_csv():
    df = pd.DataFrame(columns=COLS)

    for d in os.listdir(DATASET_PATH):
        s = './' + DATASET_PATH + '/' + d + '/' + d + '_accelerometer.log'
        print(s)
        df = df.append(process_file(s), ignore_index=True)

    # Normalize
    df[NORM_COLS] = df[NORM_COLS].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    df.to_csv('proc.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)

def process_wrapper(lines, fname):
    # Extract raw sensor values
    sensorVals = []
    with open(fname, 'r') as f:
        for line in lines:
            vals = line.split()
            try:
                sensorVals.append((float(vals[1]), float(vals[2]), float(vals[3])))
            except:
                pass

    results = []

    # Mean sensor values
    means = ft._avg(sensorVals)
    results.extend([means[0], means[1], means[2]])

    # Stddev
    results.extend(ft._stddev(sensorVals, means))

    # Absdev
    results.extend(ft._absdev(sensorVals, means))

    # Resultant
    results.append(ft._resultant(sensorVals))

    # Zero crossings
    results.extend(ft._zerocross(sensorVals))

    # Minima
    results.extend(ft._minima(sensorVals))

    # Maxima
    results.extend(ft._maxima(sensorVals))

    # Class
    results.append(os.path.basename(fname).split('_')[0])

    # Histograms
    results.extend(ft._hist(sensorVals, -1.5 * g, 1.5 * g))

    return results

def chunkify(fname, lines=LINES_CHUNK):
    with open(fname, 'r') as f:
        # Skip csv header
        next(f)

        # Chunkify
        for chunk in zip(*[f] * lines):
            yield chunk

def process_file(file):
    # Create dataframe that stores the results
    df = pd.DataFrame(columns=COLS)

    pool = mp.Pool(processes=None)
    jobs = []

    # Create jobs
    for chunk in chunkify(file, LINES_CHUNK):
        jobs.append(pool.apply_async(process_wrapper, args=(chunk,file,)))

    # Wait for all jobs to finish
    for job in range(len(jobs)):
        df.loc[df.shape[0]] = jobs[job].get()

    pool.close()

    return df