import click
import pandas as pd
from transnalize.maestro import Maestro


@click.command()
@click.argument('file', type=click.File('r'))
@click.argument('output_path', type=click.Path())
@click.argument('output_name', type=click.STRING)
@click.option('-b', '--batch', type=click.INT, default=10, help='How many rows processed at a time')
@click.option('-t', '--threads', type=click.INT, default=1, help='Number of worker threads')
def cli(file, output_path, output_name, batch, threads):
    '''
    Translate and analyze sentiment in batch mode
    '''

    click.echo('Initializing...')
    df = pd.read_csv(file.name)
    maestro = Maestro(df, output_path, output_name, batch)
    maestro.play(n_thread=threads)
