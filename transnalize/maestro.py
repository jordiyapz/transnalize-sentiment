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
import time

ERR_STR = '{}: Error {}.\n'


class Maestro:
    def __init__(self, df, output_path, output_name, batch):
        # storing variables
        self.df = df
        self.filename = Path(output_path) / output_name
        self.raw_file = '{}_raw.csv'.format(self.filename)
        self.batch = batch

        # initialize tools
        self.translator = Translator()
        self.__initialize_senti()

        # collect jobs
        job_list = self.__collect_jobs()
        self.total_job = len(job_list)

        # initialize queues
        self.jobs = Queue(maxsize=self.total_job)
        for job in job_list:
            self.jobs.put(job)
        self.result = Queue(maxsize=self.total_job)

        # setup threading variables
        self.stop = threading.Event()
        self.worker_ct_lock = threading.Lock()
        self.worker_ct = 0  # num_of_spawned worker

    def __initialize_senti(self):
        self.senti = PySentiStr()
        self.senti.setSentiStrengthPath(
            str(Path.cwd()/'lib'/'SentiStrengthCom.jar'))
        self.senti.setSentiStrengthLanguageFolderPath(str(Path.cwd()/'lang'))

        # simple test to make sure senti works
        test = self.senti.getSentiment(['You are beautiful'], 'dual')
        assert type(test) is list
        assert type(test[0]) is tuple

    def __collect_jobs(self):
        try:
            out_df = pd.read_csv(self.raw_file, header=None)
            processed_ser = self.df['tweetid'].isin(out_df[1])
        except FileNotFoundError:
            zeros = np.zeros((len(self.df.index),), dtype=bool)
            processed_ser = pd.Series(zeros)

        job_list = processed_ser[~processed_ser].index
        job_list = list(grouper(job_list, self.batch))
        if len(job_list) > 0:
            job_list[-1] = tuple(job for job in job_list[-1]
                                 if job is not None)

        return job_list

    def __get_sentiment(self, series, score='dual'):
        try:
            items = series.to_numpy().tolist()
            if len(series.index) == 1:
                translations = [self.translator.translate(items)]
            else:
                translations = self.translator.translate(items)
        except Exception as e:
            print(ERR_STR.format('get_sentiment', 'in languange translate'), e)
            return

        texts = [tr.text for tr in translations]
        langs = [tr.src for tr in translations]

        try:
            sentis = self.senti.getSentiment(texts, score)
        except Exception as e:
            print(ERR_STR.format('get_sentiment', 'in getting sentiment'), e)
            return

        return [(*st, lang, '"{}"'.format(text.replace('"', '\"')))
                for st, lang, text in zip(sentis, langs, texts)]

    def __transnalize(self, thread_num):
        with self.worker_ct_lock:
            self.worker_ct = self.worker_ct + 1
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

            result = [(j, id, *senti)
                      for j, id, senti in zip(job, df.iloc[:, 0], senti_batch)]
            self.result.put(result)

    def __save(self):
        total_batch = int(np.ceil(len(self.df.index)/self.batch))
        pbar = tqdm(total=total_batch, initial=(total_batch - self.total_job))
        while not self.stop.is_set() or not self.result.empty():
            if not self.result.empty():
                try:
                    with open(self.raw_file, 'a', encoding='utf-8') as f:
                        while not self.result.empty():
                            results = self.result.get()
                            for result in results:
                                res = (*map(str, result[:-1]), result[-1])
                                f.write(','.join(res)+'\n')
                            pbar.update(1)
                except Exception as e:
                    print(ERR_STR.format('save', 'writing file'), e)
                    break
        print('Rebuilding...')
        self.__rebuild()
        print('Closing...')
        pbar.close()

    def __rebuild(self):
        try:
            sf = pd.read_csv(self.raw_file, header=None, names=[
                             'order', 'tweetid', '+', '-', 'src_lang', 'translation'])
            sf.sort_values('order', inplace=True)
            sf.to_csv('{}.csv'.format(self.filename), index=None)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(ERR_STR.format('rebuild', 'on rebuilding csv'), e)

    def play(self, n_thread=1):
        if n_thread < 1:
            return
        with ThreadPoolExecutor(max_workers=n_thread+1) as executor:
            try:
                executor.map(self.__transnalize, range(n_thread))
                print('Spawing {} workers...'.format(n_thread))
                while self.worker_ct is 0:
                    pass  # waiting for any worker being spawned
                print('Aye, Sir!')
                executor.submit(self.__save)

                # as long as there are a job and atleast a worker
                while not self.jobs.empty() and self.worker_ct > 0:
                    # wait for any keyboard interrupt
                    time.sleep(.5)  # power napping for half second
                # either no job left or all worker has been despawned
                self.stop.set()

                if self.jobs.empty():
                    print('All done!')
                if self.worker_ct is 0:
                    print('All workers quit their job!')
            except KeyboardInterrupt:
                print('\nKeyboard interrupt')
            except Exception as e:
                print(ERR_STR.format('play', 'something went wrong'), e)
            finally:
                self.stop.set()

        print('Byee 👋')
