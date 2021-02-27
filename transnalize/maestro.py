import pandas as pd
import numpy as np
from queue import Queue
from sentistrength import PySentiStr
from pygoogletranslation import Translator
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
from tqdm import tqdm
import time
import csv
from transnalize.itertools_recipes import grouper

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
        self.results = Queue(maxsize=self.total_job)

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

    def __despawn_worker(self):
        with self.worker_ct_lock:
            self.worker_ct = self.worker_ct - 1

    def __translate(self, thread_num):
        with self.worker_ct_lock:
            self.worker_ct = self.worker_ct + 1
        while not self.stop.is_set() and not self.jobs.empty():
            job = self.jobs.get()
            try:
                mini_df = self.df.loc[job, ]  # trailing comma is needed
                ids = mini_df.iloc[:, 0]
                items = mini_df.iloc[:, -1].to_numpy().tolist()
            except Exception as e:
                print('Worker #{} got pandas error: {}'.format(thread_num, e))
                break

            try:
                if len(items) == 1:
                    translations = [self.translator.translate(items)]
                else:
                    translations = self.translator.translate(items)
            except Exception as e:
                print('Worker #{} got translation error: {}'.format(thread_num, e))
                break

            self.results.put((job, ids, translations))

        self.__despawn_worker()

    def __save(self, results):
        with open(self.raw_file, 'a', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(results)

    def __process(self, score='dual'):
        total_batch = int(np.ceil(len(self.df.index)/self.batch))
        pbar = tqdm(total=total_batch, initial=(total_batch - self.total_job))

        while not self.stop.is_set() or not self.results.empty():
            time.sleep(2)
            if not self.results.empty():
                # merges all results
                job_list, id_list, translation_list = ([], [], [])
                steps = 0
                while not self.results.empty():
                    job, ids, translations = self.results.get()
                    job_list.extend(job)
                    id_list.extend(ids)
                    translation_list.extend(translations)
                    steps = steps + 1

                # analyze sentiments
                texts = [tr.text for tr in translation_list]
                try:
                    sentis = self.senti.getSentiment(texts, score)
                except Exception as e:
                    print('Process got sentistrength error:', e)
                    break

                try:
                    rows = [(order, i, *senti, tr.src, text)
                            for order, i, senti, tr, text in
                            zip(job_list, id_list, sentis, translation_list, texts)]
                except Exception as e:
                    print(e)
                    break

                try:
                    self.__save(rows)
                except Exception as e:
                    print('Process got on save error:', e)
                    break

                pbar.update(steps)
            time.sleep(.1)  # prevent too much loop checking

        if not self.stop.is_set():
            self.stop.set()  # force stop all threads

        print('Rebuilding...')
        self.__rebuild()

        print('Exiting...')
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
                executor.map(self.__translate, range(n_thread))
                print('Spawing {} workers...'.format(n_thread))
                while self.worker_ct is 0:
                    pass  # waiting for any worker being spawned
                print('Aye, Sir!')
                executor.submit(self.__process)

                # as long as there are atleast a worker
                while self.worker_ct > 0:
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
