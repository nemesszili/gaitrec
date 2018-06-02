import pandas as pd
import numpy as np

import time
import os

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from util.process import process_file, export_csv
from util.const import CSV_PATH, SEED, TEST_RATIO, BG_SIZE
from util.visualize import plot_AUC, plot_scores

def main():
    if not os.path.exists(CSV_PATH):
        start_time = time.time()
        export_csv()
        print("--- %s seconds ---" % (time.time() - start_time))

    # Import data
    df = pd.read_csv(CSV_PATH)

    users = ['u%03d' % i for i in range(1, 51)]
    background = users[-BG_SIZE:]
    X_background = df.loc[df['class'].isin(background)]
    X_background = X_background.drop(X_background[['class']], axis=1)

    # Train background model
    gmm_bg = GaussianMixture()
    gmm_bg.fit(X_background)

    scores = pd.DataFrame(columns=['score', 'label', 'class'])

    for u in users[:-BG_SIZE]:
        # Select data for current user
        user_data = df[df['class'] == u]
        x = user_data.drop(user_data[['class']], axis=1)

        X_train, X_test = train_test_split(x, random_state=SEED)

        gmm = GaussianMixture()
        gmm.fit(X_train)

        # Collect log-likelihood scores
        pos_scores = pd.DataFrame(columns=['score', 'label', 'class'])
        pos_scores['score'] = gmm.score_samples(X_test)
        pos_scores['label'] = '1'
        pos_scores['class'] = u
        scores = scores.append(pos_scores)

        neg_scores = pd.DataFrame(columns=['score', 'label', 'class'])
        neg_scores['score'] = gmm_bg.score_samples(X_test)
        neg_scores['label'] = '0'
        neg_scores['class'] = u
        scores = scores.append(neg_scores)

        plot_scores(u, scores)

    plot_AUC(scores)

if __name__ == '__main__':
    main()
