import pandas as pd
from typing import List, Dict, Any, Optional
from src.features import entities
from src.data import datasets
from pathlib import Path
import datetime
from datetime import datetime
import scipy.stats
import logging
import numpy as np


project_dir = Path(__file__).resolve().parents[2]


def _age_gender_population(age_gender_df: pd.DataFrame) -> pd.DataFrame:
    """Polish census data gives age and gender together. For each age (or age range) there is a number of males and
    females provided. This function generates a dataframe of people with age and gender. The length of the dataframe
    equals the total number of people in the Census data. """
    ages = []
    genders = []
    for idx, row in age_gender_df.iterrows():
        ages.extend([row.Age] * row.Total)
        genders.extend([entities.Gender.MALE.value] * row.Males)
        genders.extend([entities.Gender.FEMALE.value] * row.Females)
    return pd.DataFrame(data={entities.prop_age: ages, entities.prop_gender: genders})


def nodes_to_dataframe(nodes: List[entities.Node]) -> pd.DataFrame:
    """A utility function that takes a list of dictionaries (here, specifically the subclass of dictionary - Node),
    converts these dictionaries into lists and creates a dataframe out of them. """
    return pd.DataFrame(data=_list_of_dicts_to_dict_of_lists(nodes))


def _list_of_dicts_to_dict_of_lists(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """A utility function that given a list of dictionaries converts them into a dictionary of named
    (by a key) lists. """
    return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}


def prepare_simulations_folder(simulations_folder: Path = None):
    """Creates a parent folder for generated population. """
    if simulations_folder is None:
        simulations_folder = project_dir / 'data' / 'simulations' / datetime.now().strftime('%Y%m%d_%H%M')
    if not simulations_folder.is_dir():
        simulations_folder.mkdir()
    return simulations_folder


def get_distribution(distribution):
    if isinstance(distribution, str):
        return getattr(scipy.stats, distribution)
    raise ValueError(f'Expected the name of a distribution, but got {distribution}')


def sample_from_distribution(sample_size, distribution_name, *args, **kwargs):
    distribution = get_distribution(distribution_name)
    return distribution.rvs(*args, size=sample_size, **kwargs)


def drop_obsolete_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns.tolist()
    to_drop = [col for col in columns if col not in entities.columns]
    return df.drop(columns=to_drop)


def age_range_to_age(df: pd.DataFrame) -> pd.DataFrame:
    idx = df[df.age.str.len() > 2].index.tolist()
    df.loc[idx, 'age'] = df.loc[idx].age.str.slice(0, 2).astype(int)
    df.loc[idx, 'age'] += np.random.choice(list(range(0, 5)), size=len(idx))
    df.age = df.age.astype(int)  # make the whole column as int
    return df


def fix_homeless(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[entities.prop_household] != -1]


def cleanup(df: pd.DataFrame) -> pd.DataFrame:
    return age_range_to_age(drop_obsolete_columns(fix_homeless(df)))


def get_age_gender_df(data_folder: Path, sheet_name: Optional[str] = datasets.age_gender_xlsx.sheet_name) \
        -> pd.DataFrame:
    age_gender_df = pd.read_excel(str(data_folder / datasets.age_gender_xlsx.file_name), sheet_name=sheet_name)
    age_gender_df['Total'] = age_gender_df['Total'].astype(int)
    age_gender_df['Males'] = age_gender_df['Males'].astype(int)
    age_gender_df['Females'] = age_gender_df['Females'].astype(int)
    try:
        age_gender_df['Age'] = age_gender_df['Age'].astype(int)
    except ValueError:
        logging.warning('Age column contains invalid literal and cannot be cast to int')
    age_gender_df['female_probability'] = age_gender_df.Females / (age_gender_df.Females + age_gender_df.Males)
    return age_gender_df


def get_age_gender_df_with_generations(data_folder: Path,
                                       sheet_name: Optional[str] = datasets.age_gender_xlsx.sheet_name) \
        -> pd.DataFrame:
    generations_df = pd.read_excel(str(data_folder.parent / datasets.generations_xlsx.file_name))
    return pd.merge(get_age_gender_df(data_folder, sheet_name), generations_df, left_on='Age', right_on='age')

