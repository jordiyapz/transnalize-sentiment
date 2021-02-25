import pandas as pd
import numpy as np
from queue import Queue
from sentistrength import PySentiStr
from pygoogletranslation import Translator
from pathlib import Path
from transnalize.itertools_recipes import grouper
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
from tqdm import tqdm

ERR_STR = '{}: Error {}.\n'


class Maestro:
    def __init__(self, df, output_path, output_name, batch):
        self.df = df
        self.output_path = output_path
        self.output_name = output_name
        self.filename = '{}.csv'.format(
            Path(self.output_path) / self.output_name)
        self.batch = batch

        # initialize translator
        self.translator = Translator()

        # initialize senti
        self.senti = PySentiStr()
        self.senti.setSentiStrengthPath(
            str(Path.cwd()/'lib'/'SentiStrengthCom.jar'))
        self.senti.setSentiStrengthLanguageFolderPath(str(Path.cwd()/'lang'))

        # simple test to make sure senti works
        test = self.senti.getSentiment(['You are beautiful'], 'dual')
        assert type(test) is list
        assert type(test[0]) is tuple

        try:
            out_df = pd.read_csv(self.filename, header=None)
            processed_ser = df['tweetid'].isin(out_df[0])
        except FileNotFoundError:
            zeros = np.zeros((len(df.index),), dtype=bool)
            processed_ser = pd.Series(zeros)
        except Exception as e:
            print(e)

        job_list = processed_ser[~processed_ser].index
        job_list = list(grouper(job_list, batch))
        job_list[-1] = tuple(job for job in job_list[-1] if job is not None)
        self.total_job = len(job_list)

        # initialize job queue
        self.jobs = Queue(maxsize=self.total_job)
        for job in job_list:
            self.jobs.put(job)

        self.result = Queue(maxsize=self.total_job)

        self.stop = threading.Event()

    def __get_sentiment(self, series, score='dual'):
        try:
            items = series.to_numpy().tolist()
            if len(series.index) == 1:
                translations = [self.translator.translate(items)]
            else:
                translations = self.translator.translate(items)
        except Exception as e:
            print(ERR_STR.format('get_sentiment', 'in languange translate'), e)

        texts = [tr.text for tr in translations]
        langs = [tr.src for tr in translations]

        try:
            sentis = self.senti.getSentiment(texts, score)
        except Exception as e:
            print(ERR_STR.format('get_sentiment', 'in getting sentiment'), e)
            return

        return [(*st, lang, '"{}"'.format(text))
                for st, lang, text in zip(sentis, langs, texts)]

    def __transnalize(self, thread_num):
        while not self.stop.is_set() and not self.jobs.empty():
            job = self.jobs.get()
            try:
                # trailing comma is needed coz job is array
                df = self.df.loc[job, ]
            except Exception as e:
                print(ERR_STR.format('transnalize', 'slicing DataFrame'), e)
                break
            try:
                senti_batch = self.__get_sentiment(df.iloc[:, -1])
            except Exception as e:
                print(ERR_STR.format('transnalize', 'getting sentiment'), e)
                break
            result = [(id, *senti)
                      for id, senti in zip(df.iloc[:, 0], senti_batch)]
            self.result.put(result)

    def __save(self):
        total_batch = int(np.ceil(len(self.df.index)/self.batch))
        pbar = tqdm(total=total_batch, initial=(total_batch - self.total_job))
        while not self.stop.is_set() or not self.result.empty():
            if not self.result.empty():
                with open(self.filename, 'a') as f:
                    while not self.result.empty():
                        result = self.result.get()
                        for res in result:
                            f.write(','.join(map(str, res))+'\n')
                        pbar.update(1)
        print('Closing...')
        pbar.close()

    def play(self, n_thread=1):
        if n_thread < 1:
            return
        print('Spawing {} workers...'.format(n_thread))
        with ThreadPoolExecutor(max_workers=n_thread+1) as executor:
            try:
                print('Start')
                executor.map(self.__transnalize, range(n_thread))
                executor.submit(self.__save)
                while True:
                    # wait for any keyboard interrupt
                    pass
            except KeyboardInterrupt:
                self.stop.set()
                print('\nKeyboard interrupt')
            except Exception as e:
                print(ERR_STR.format('play', 'something went wrong'))
