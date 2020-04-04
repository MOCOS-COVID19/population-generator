# population-generator
## About 
This project generates a population of Poland, and of some Polish cities in particular, based on Polish National Census data from 2011 and other, usually more recent, statistics provided by GUS (Statistics Poland), city councils and other data providers. 

## Installation
The code requires Python >= 3.7. To install dependencies execute:
```
pip install -r requirements.txt
```

### Unit tests
To run  unit tests, executeL 
```
python -m unittest discover
```

### Population generation
To generate a population run one of the following scripts:
* `src.generation.population_generator_for_cities` - if you wish to generate a population of a city, currently supported are Warsaw (data subfolder name: WW) and Wrocław (data subfolder name: DW). 
* `src.generation.population_generator_for_poland` - if you with to generate a population of Poland 


## Useful links
### Technical stuff:
* [MOCOS slack space](https://modellingncov2019.slack.com/)
* [Jira board for programming team](https://mocos-covid19.atlassian.net/secure/RapidBoard.jspa?rapidView=1&projectKey=MC&view=planning&selectedIssue=MC-12&issueLimit=100)
### Tracking the epidemics
* [Timeline of the 2019 Wuhan coronavirus outbreak](https://en.wikipedia.org/wiki/Timeline_of_the_2019%E2%80%9320_Wuhan_coronavirus_outbreak)
* [Tracking coronavirus: Map, data and timeline](https://bnonews.com/index.php/2020/02/the-latest-coronavirus-cases/)

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── simulations    <- The output, generated data.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    ├── notebooks          <- Jupyter notebooks. 
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── data           <- Scripts to describe the data
    │   ├── generations    <- Scripts to generate populations
    │   ├── preprocessing  <- Scripts to further process raw or processed data
    │   └── validation     <- Scripts to perform sanity checks on generated population.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

### Git Large File Storage
[git lfs](https://git-lfs.github.com/) should be used to store big files.
Please follow [instructions](https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage) to set up git-lfs on your end.
Check `.gitattributes` for file extensions and location currently tracked with git-lfs. 

If you need to track different paths, please add them using `git lfs track [path-to-be-tracked]`.
This will append new lines to `.gitattributes` file. Otherwise, you can amend paths in the file by hand.
