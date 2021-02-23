import setuptools

setuptools.setup(
    name='transnalize-sentiment',
    version='0.1.0',
    author='Jordi Yaputra',
    author_email='jordiyaputra@gmail.com',
    description='Batch translate and analyze sentiment strength',
    py_module=['transnalize'],
    install_requires=['pandas', 'numpy',
                      'pygoogletranslation', 'sentistrength', 'tqdm', 'click'],
    entry_points='''
    [console_scripts]
    transnalize=transnalize:cli
    '''
)
