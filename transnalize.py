import click
import pandas as pd
from sentistrength import PySentiStr
from pygoogletranslation import Translator
from pathlib import Path
from utils import transnalize


@click.command()
@click.argument('file', type=click.File('r'))
@click.argument('output', type=click.File('w'))
@click.option('-b', '--batch', type=click.INT, default=100, help='How many rows processed at a time')
@click.option('-s', '--start', type=click.INT, default=None, help='Row to start with')
@click.option('-e', '--end', type=click.INT, default=None)
def cli(file, output, batch, start, end):
    '''
    Translate and analyze sentiment in batch mode
    '''
    df = pd.read_csv(file.name)

    translator = Translator()

    senti = PySentiStr()
    senti.setSentiStrengthPath(
        str(Path.cwd()/'lib'/'SentiStrengthCom.jar'))
    senti.setSentiStrengthLanguageFolderPath(str(Path.cwd()/'lang'))

    transnalize(translator, senti, df, output.name, batch, start, end)
