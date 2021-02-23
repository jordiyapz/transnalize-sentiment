import pandas as pd
import numpy as np
from tqdm import tqdm


def get_sentiment(translator, senti, series, score='scale'):
    if len(series.index) == 1:
        descriptions = translator.translate(series.to_numpy().tolist()).text
    else:
        descriptions = [tr.text for tr in translator.translate(
            series.to_numpy().tolist())]
    return senti.getSentiment(descriptions, score)


def transnalize(translator, senti, df, csv_path, batch=100, start=None, end=None):
    last = min(len(df.index), end) if end else len(df.index)

    try:
        sentiment_df = pd.read_csv(csv_path, names=['id', 'pos', 'neg'])
        btm = len(sentiment_df.index)
    except FileNotFoundError:
        btm = 0

    btm = max(btm, start) if start else btm

    ranges = np.arange(btm+batch, last+batch, batch)
    if len(ranges) == 0:
        return 'Done'
    with open(csv_path, 'a') as f:
        for i in tqdm(ranges, desc='analyzing', position=0, leave=True):
            top = i if i < last else last  # constrain top

            sentimen_batch = get_sentiment(
                translator, senti, df.iloc[btm:top, -1], score='dual')

            # save chunk
            for idx, sentimen in zip(df.iloc[btm:top, 0], sentimen_batch):
                f.write('{},{},{}\n'.format(idx, sentimen[0], sentimen[1]))

            btm = top
