import pandas as pd

import multiprocessing as mp
import os

from itertools import izip, islice
from const import LINES_CHUNK, g
import features as ft

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
    results.append('u001')

    # Histograms
    results.extend(ft._hist(sensorVals, -1.5 * g, 1.5 * g))

    return results

def chunkify(fname, lines=LINES_CHUNK):
    with open(fname, 'r') as f:
        # Skip csv header
        next(f)

        # Chunkify
        for chunk in izip(*[f] * lines):
            yield chunk

def process_file(file):
    # Features to be extracted
    cols = ['xavg', 'yavg', 'zavg', 'xstddev', 'ystddev', 'zstddev', 'xabsdev', 'yabsdev', 'zabsdev', 
            'res', 'xzcr', 'yzcr', 'zzcr', 'minx', 'miny', 'minz', 'maxx', 'maxy', 'maxz', 'class',
            'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
            'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9',
            'z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9']

    # Create dataframe that stores the results
    df = pd.DataFrame(columns=cols)

    pool = mp.Pool(processes=None)
    jobs = []

    # Create jobs
    for chunk in chunkify(file, LINES_CHUNK):
        jobs.append(pool.apply_async(process_wrapper, args=(chunk,file,)))

    # Wait for all jobs to finish
    for job in range(len(jobs)):
        res = jobs[job].get()
        print(res)
        print(len(res))
        print(df.shape[1])
        df.loc[df.shape[0]] = res

    pool.close()

    print(df.head())