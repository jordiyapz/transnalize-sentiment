from setuptools import setup, find_packages

setup(
    name='transnalize-sentiment',
    version='1.0.0',
    author='Jordi Yaputra',
    author_email='jordiyaputra@gmail.com',
    description='Batch translate and analyze sentiment strength',
    packages=find_packages(),
    license='MIT',
    py_module=['transnalize'],
    classifiers =(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=['pandas', 'numpy',
                      'pygoogletranslation', 'sentistrength', 'tqdm', 'click'],
    entry_points={
        'console_scripts': ['transnalize=transnalize.cli:cli']
    }
)
