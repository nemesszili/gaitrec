import pandas as pd
import numpy as np

import time
import os

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from util.process import process_file, export_csv
from util.const import CSV_PATH, SEED, TEST_RATIO
from util.visualize import plotAUC

def main():
    if not os.path.exists(CSV_PATH):
        export_csv()

    # Import data
    df = pd.read_csv(CSV_PATH)

    users = ['u%03d' % i for i in range(1, 51)]
    background = users[-10:]
    X_background = df.loc[df['class'].isin(background)]
    X_background = X_background.drop(X_background[['class']], axis=1)

    scores = pd.DataFrame(columns=['score', 'label'])

    for u in users[:-10]:
        # Select data for current user
        user_data = df[df['class'] == u]

        x = user_data.drop(user_data[['class']], axis=1)

        X_train, X_test = train_test_split(x, random_state=SEED)

        gmm = GaussianMixture()
        gmm.fit(X_train)

        # Collect log-likelihood scores
        pos_scores = pd.DataFrame(columns=['score', 'label'])
        pos_scores['score'] = gmm.score_samples(X_test)
        pos_scores['label'] = '1'
        scores = scores.append(pos_scores)

        neg_scores = pd.DataFrame(columns=['score', 'label'])
        neg_scores['score'] = gmm.score_samples(X_background)
        neg_scores['label'] = '0'
        scores = scores.append(neg_scores)

    plotAUC(scores)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))