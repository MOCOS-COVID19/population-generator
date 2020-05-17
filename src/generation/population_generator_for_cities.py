import logging
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src.data import datasets, entities
from src.features import (Feature, FeatureParams, SocialCompetence, SocialCompetenceParams, EmploymentParams,
                          Employment)
from src.generation.population_generator_common import (age_gender_population,
                                                        sample_from_distribution,
                                                        rename_index, drop_obsolete_columns,
                                                        age_range_to_age)
from src.preprocessing import preprocessing_poland


def process_other_features(population: pd.DataFrame, other_features: Optional[List[Tuple[Feature, FeatureParams]]])\
        -> pd.DataFrame:
    """
    Updates individual in a population with additional features
    :param population: population dataframe
    :param other_features: list additional features with their parameters
    :return: updated population dataframe
    """
    if other_features is None:
        return population
    for feature, feature_params in other_features:
        population = feature.generate(feature_params, population)
    return population


def narrow_housemasters_by_headcount_and_age_group(household_by_master, household_row):
    """
    Przy wyborze reprezentanta kierowano się następującymi zasadami:
    •   gdy mieszkanie zajmowała tylko jedna osoba, ona była reprezentantem;
    •   gdy  mieszkanie  zajmowali  małżonkowie/partnerzy  z  dziećmi  lub  bez  dzieci,
    reprezentantem należało wybrać jednego z małżonków/partnerów,
    •   gdy mieszkanie zajmował rodzic z dzieckiem/dziećmi, reprezentantem był rodzic;
    •   gdy  mieszkanie  zajmowała  rodzina  trzypokoleniowa  –  należało  wybrać  osobę
    ze średniego pokolenia;
    •   gdy  żaden  z  powyższych  warunków  nie  był  spełniony,  reprezentantem  mogła  zostać
    osoba dorosła mieszkająca w tym mieszkaniu.
    :param household_by_master:
    :param household_row:
    :return:
    """

    elderly = household_row.elderly == 1
    middle = household_row.middle == 1
    young = household_row.young == 1
    masters = household_by_master[household_by_master.Headcount == household_row.household_headcount]

    if elderly and not middle and not young:
        return masters[masters.elderly == 1]
    if not elderly and middle and not young:
        return masters[masters.middle == 1]
    if not elderly and not middle and young:
        return masters[masters.young == 1]

    if not elderly:
        masters = masters[masters.elderly == 0]
    if not middle:
        masters = masters[masters.middle == 0]
    if not young:
        masters = masters[masters.young == 0]

    if household_row.family_type == 0 or household_row.family_type == 3:
        # does it have different probability (given we are restricted on other conditions below?)
        # headcount agreement plus all ages that are present in the household
        # single person households also falls here
        return masters

    if household_row.household_headcount == 2:
        # the only left here are family_type == 1
        # choose the oldest generation
        if elderly:
            return masters[masters.elderly == 1]
        return masters[masters.middle == 1]
        # young only covered at the beginning

    if household_row.household_headcount >= 3 and household_row.family_type == 1:
        # family_type == 1
        if household_row.relationship == 'Bez osób spoza rodziny':  # MFC+, FCC+, MCC+
            if elderly and middle and young:
                # choose middle or elderly - parents are various age?
                return masters[masters.young == 0]
            elif elderly and middle and not young:
                # choose elderly, as the child is middle
                return masters[masters.elderly == 1]
            elif not elderly and middle and young:
                # choose middle
                return masters[masters.middle == 1]
            elif elderly and not middle and young:
                # choose elderly
                return masters[masters.elderly == 1]
        elif household_row.relationship == 'Z innymi osobami':
            # basically any adult
            # MF+O or MC+O or FC+O - other can be any age
            return masters
        else:  # Z krewnymi w linii prostej starszego pokolenia
            if household_row.house_master == 'członek rodziny':
                if elderly and middle:
                    return masters[masters.middle == 1]
                elif (elderly or middle) and young:
                    return masters[masters.young == 1]
            elif household_row.house_master == 'krewny starszego pokolenia':
                if elderly and (middle or young):
                    return masters[masters.elderly == 1]
                elif middle and young:
                    return masters[masters.middle == 1]
            else:  # inna osoba
                return masters

    if household_row.household_headcount >= 4 and household_row.family_type == 2:
        if household_row.relationship == 'Spokrewnione w linii prostej':
            if household_row.house_master == 'członek rodziny młodszego pokolenia ':
                if elderly and middle:
                    return masters[masters.middle == 1]
                elif (elderly or middle) and young:
                    return masters[masters.young == 1]
            elif household_row.house_master == 'członek rodziny starszego pokolenia':
                if elderly and (middle or young):
                    return masters[masters.elderly == 1]
                elif middle and young:
                    return masters[masters.middle == 1]
            else:  # inna osoba
                return masters
        elif household_row.relationship == 'Niespokrewnione w linii prostej':
            return masters

    raise ValueError(f'Couldn\'t find masters for {household_row}')


def _get_family_structure(families_and_children_df, headcount):
    df_fc = families_and_children_df.loc[families_and_children_df.nb_of_people == headcount]
    try:
        final_structure_idx = np.random.choice(df_fc.index.to_list(),
                                               p=df_fc.prob_with_young_adults_per_headcount)
        return df_fc.loc[final_structure_idx, 'structure']
    except ValueError:
        logging.exception(f'Something went wrong for {headcount}')


def generate_households(data_folder: Path, output_folder: Path, population_size: int, start_index: int = 0) \
        -> pd.DataFrame:
    """
    Given a population size and the path to a folder with data, generates households for this population.
    :param data_folder: path to a folder with data
    :param voivodship_folder: path to a folder with voivodship data
    :param output_folder: path to a folder where housedhols should be saved
    :param population_size: size of a population to accommodate
    :param start_index: index to start the numbering of households with
    :return: a pandas dataframe with households to lodge the population.
    """
    households = None
    households_ready_feather = output_folder / datasets.output_households_interim_feather.file_name
    try:
        households = pd.read_feather(str(households_ready_feather))
    except Exception as e:
        logging.warning("Reading interim Feather failed, reason: " + str(e))

    if households is None:
        logging.warning("Recomputing...")
        try:
            households = pd.read_feather(str(output_folder / datasets.output_households_basic_feather.file_name))
        except Exception as e:
            logging.warning("Reading basic Feather failed, reason: " + str(e))
            households = preprocessing_poland.generate_household_indices(data_folder, output_folder, population_size,
                                                                         start_index)

        masters_age = []
        masters_gender = []

        # household master
        household_by_master = pd.read_excel(
            str(data_folder / datasets.voivodship_cities_households_by_master_xlsx.file_name),
            sheet_name=datasets.voivodship_cities_households_by_master_xlsx.sheet_name)

        for idx, household_row in tqdm(households.iterrows(), total=len(households)):
            masters = narrow_housemasters_by_headcount_and_age_group(household_by_master, household_row)
            p = masters['Count'] / masters['Count'].sum()
            index = np.random.choice(masters.index.tolist(), p=p)
            masters_age.append(masters.loc[index, 'Age'])
            masters_gender.append(entities.gender_from_string(masters.loc[index, 'Gender']).value)

        households['master_age'] = masters_age
        households['master_gender'] = masters_gender
        try:
            households.to_feather(str(households_ready_feather))
        except Exception as e:
            logging.warning("Saving interim Feather failed, reason: " + str(e))

    return households


def assign_house_masters(households, population):
    # get indices of households of a specific age, gender
    df23 = households.groupby(by=['master_age', 'master_gender'], sort=False).size() \
        .reset_index().rename(columns={0: 'total'})

    households[entities.h_prop_house_master_index] = entities.HOUSEHOLD_NOT_ASSIGNED
    households[entities.h_prop_inhabitants] = '[]'

    aged_19_and_less = ('18', '19')
    aged_20_24 = ('20', '21', '22', '23', '24')
    aged_25_29 = ('25', '26', '27', '28', '29')
    unassigned_households = defaultdict(list)

    # given a household and its master's age and gender
    # select a person
    # set index of a person onto a household
    # set index of a household onto a person
    for idx, df23_row in tqdm(df23.iterrows(), desc='Master selection'):
        if df23_row.master_age == '19 lat i mniej':
            subpopulation = population[population[entities.prop_age].isin(aged_19_and_less)
                                       & (population[entities.prop_gender] == df23_row['master_gender'])].index.tolist()
        elif df23_row.master_age == '20-24':
            subpopulation = population[population[entities.prop_age].isin(aged_20_24)
                                       & (population[entities.prop_gender] == df23_row['master_gender'])].index.tolist()
        elif df23_row.master_age == '25-29':
            subpopulation = population[population[entities.prop_age].isin(aged_25_29)
                                       & (population[entities.prop_gender] == df23_row['master_gender'])].index.tolist()
        else:
            subpopulation = population[(population[entities.prop_age] == df23_row['master_age'])
                                       & (population[entities.prop_gender] == df23_row['master_gender'])].index.tolist()
        households_indices = households[(households.master_age == df23_row.master_age) &
                                        (households.master_gender == df23_row.master_gender)].index.tolist()
        try:
            masters_indices = np.random.choice(subpopulation, replace=False, size=df23_row.total)
        except ValueError as e:
            if str(e) == 'Cannot take a larger sample than population when \'replace=False\'':
                logging.info(f'THere are more masters than people in the population for {df23_row}. '
                             f'Making all people within this cluster masters.')
                masters_indices = subpopulation
                covered_indices = np.random.choice(households_indices, replace=False, size=len(masters_indices))
                not_covered_indices = set(households_indices).difference(set(covered_indices))
                unassigned_households[df23_row['master_gender']].extend(not_covered_indices)
                households_indices = covered_indices
            elif str(e) == '\'a\' cannot be empty unless no samples are taken':
                logging.error(f'Population for {df23_row} is empty.')
                unassigned_households[df23_row['master_gender']].extend(households_indices)
                continue
            else:
                raise
        population.loc[masters_indices, entities.prop_household] = households_indices
        households.loc[households_indices, entities.h_prop_house_master_index] = masters_indices
        households.loc[households_indices, entities.h_prop_inhabitants] = [str([x]) for x in masters_indices]

    # we may have some unassigned households
    if len(unassigned_households) == 0:
        return
    youngsters = [str(x) for x in range(0, 18)]
    for gender, households_indices in unassigned_households.items():
        subpopulation = population[~population[entities.prop_age].isin(youngsters)
                                   & (population[entities.prop_gender] == gender)
                                   & (population[entities.prop_household] == entities.HOUSEHOLD_NOT_ASSIGNED)] \
            .index.tolist()
        masters_indices = np.random.choice(subpopulation, replace=False, size=len(households_indices))
        population.loc[masters_indices, entities.prop_household] = households_indices
        households.loc[households_indices, entities.h_prop_house_master_index] = masters_indices
        households.loc[households_indices, entities.h_prop_inhabitants] = [str([x]) for x in masters_indices]

    households[entities.h_prop_unassigned_occupants] = households[entities.h_prop_household_headcount] - 1


def age_gender_generation_population_from_files(data_folder: Path) -> pd.DataFrame:
    age_gender_df = pd.read_excel(str(data_folder / datasets.age_gender_xlsx.file_name),
                                  sheet_name=datasets.age_gender_xlsx.sheet_name)
    population = age_gender_population(age_gender_df)
    population['age'] = population['age'].astype(str)
    # add age generation for each person
    production_age_df = pd.read_excel(str(data_folder / datasets.production_age.file_name),
                                      sheet_name=datasets.production_age.sheet_name)
    production_age_df['age'] = production_age_df['age'].astype(str)
    return pd.merge(population, production_age_df, on=['age', 'gender'], how='left')


def assign_people_to_gafs(gaf, gaf_first_index, population, homeless_indices):
    gaf_count = len(gaf.index)
    gaf[entities.h_prop_household_index] = list(range(gaf_first_index, gaf_first_index + gaf_count))
    gaf[entities.h_prop_inhabitants] = '[]'
    gaf = gaf.set_index(entities.h_prop_household_index)

    gaf_prop_inhabitants_idx = gaf.columns.get_loc(entities.h_prop_inhabitants)
    population_prop_household_idx = population.columns.get_loc(entities.prop_household)

    for facility_idx, facility in gaf.iterrows():
        inhabitants = []
        age_groups = entities.to_age_groups(facility.young, facility.middle, facility.elderly)
        age_groups_permuted = random.choices(age_groups, k=facility.headcount)
        for age_group in age_groups_permuted:
            homeless_idx = homeless_indices[age_group.name].pop()
            population.iat[homeless_idx, population_prop_household_idx] = facility_idx
            inhabitants.append(homeless_idx)

        gaf.loc[facility_idx].iat[gaf_prop_inhabitants_idx] = str(inhabitants)


def generate_population(data_folder: Path, output_folder: Path,
                        other_features: Optional[List[Tuple[Feature, FeatureParams]]] = None) -> \
        Tuple[pd.DataFrame, pd.DataFrame]:
    # if population already generated, read from file
    population_ready_xlsx = output_folder / datasets.output_population_xlsx.file_name
    households_ready_xlsx = output_folder / datasets.output_households_xlsx.file_name
    if population_ready_xlsx.is_file() and households_ready_xlsx.is_file():
        return pd.read_excel(str(population_ready_xlsx)), pd.read_excel(str(households_ready_xlsx))

    population_ready_csv = output_folder / datasets.output_population_csv.file_name
    households_ready_csv = output_folder / datasets.output_households_csv.file_name
    if population_ready_csv.is_file() and households_ready_csv.is_file():
        return pd.read_csv(str(population_ready_csv)), pd.read_csv(str(households_ready_csv))

    # if not generated yet, create from scratch
    # get all population with their age and gender
    population = age_gender_generation_population_from_files(data_folder)

    # population size
    population_size = len(population.index)

    # initial household assignemtn
    population[entities.prop_household] = entities.HOUSEHOLD_NOT_ASSIGNED

    # get group accommodation facilities - GAF
    try:
        gaf = pd.read_csv(str(data_folder / datasets.social_care_houses_csv.file_name))
        gaf_headcount = gaf.headcount.sum()
    except FileNotFoundError:
        gaf = None
        gaf_headcount = 0

    # get households
    logging.info('Generating households')
    households = generate_households(data_folder, output_folder,
                                     population_size - gaf_headcount)

    logging.info('House master assignment')
    assign_house_masters(households, population)

    logging.info('Finding homeless people...')
    # now we need to lodge other people
    homeless = population[population[entities.prop_household] == entities.HOUSEHOLD_NOT_ASSIGNED]
    _homeless_indices = {'young': homeless[homeless.generation == 'young'].index.tolist(),
                         'middle': homeless[homeless.generation == 'middle'].index.tolist(),
                         'elderly': homeless[homeless.generation == 'elderly'].index.tolist()}
    for homeless_l in _homeless_indices.values():
        random.shuffle(homeless_l)

    households_prop_unassigned_occupants_idx = households.columns.get_loc(entities.h_prop_unassigned_occupants)
    households_prop_inhabitants_idx = households.columns.get_loc(entities.h_prop_inhabitants)
    population_prop_household_idx = population.columns.get_loc(entities.prop_household)

    logging.info('Assigning people to group accommodation facilities')

    if gaf is not None:
        gaf_first_index = households[entities.h_prop_household_index].max() + 1
        assign_people_to_gafs(gaf, gaf_first_index, population, _homeless_indices)

    logging.info('Selecting households with housemasters and headcount greater than 1...')
    households2 = households[(households.household_headcount > 1)
                             & (households.house_master_index != entities.HOUSEHOLD_NOT_ASSIGNED)]
    households_interim = defaultdict(list)

    try:
        for idx, household_row in tqdm(households2.iterrows(), desc='Lodging population - first assignments',
                                       total=len(households2)):
            inhabitants = [household_row.house_master_index]
            # lodged_headcount = 1
            try:
                hm_generation = population.iloc[household_row.house_master_index]['generation']
            except ValueError as e:
                logging.error(
                    f'ValueError ({str(e)}) for {household_row.household_index} '
                    f'(house_master_index={household_row.house_master_index})')
                continue
            except TypeError as e:
                logging.error(
                    f'TypeError ({str(e)}) for {household_row.household_index} '
                    f'(house_master_index={household_row.house_master_index})')
                continue
            # at least one person from each generation

            age_group = entities.AgeGroup.middle.name
            if household_row.middle == 1 and hm_generation != age_group \
                    and len(inhabitants) < household_row.household_headcount:
                try:
                    homeless_idx = _homeless_indices[age_group].pop()
                    population.iat[homeless_idx, population_prop_household_idx] = household_row.household_index
                    inhabitants.append(homeless_idx)
                except StopIteration:
                    logging.error(f'No more people within {age_group} generation')

            age_group = entities.AgeGroup.elderly.name
            if household_row.elderly == 1 and hm_generation != age_group \
                    and len(inhabitants) < household_row.household_headcount:
                try:
                    homeless_idx = _homeless_indices[age_group].pop()
                    population.iat[homeless_idx, population_prop_household_idx] = household_row.household_index
                    inhabitants.append(homeless_idx)
                except StopIteration:
                    logging.error(f'No more people within {age_group} generation')

            age_group = entities.AgeGroup.young.name
            if household_row.young == 1 and hm_generation != age_group \
                    and len(inhabitants) < household_row.household_headcount:

                try:
                    homeless_idx = _homeless_indices[age_group].pop()
                    population.iat[homeless_idx, population_prop_household_idx] = household_row.household_index
                    inhabitants.append(homeless_idx)
                except StopIteration:
                    logging.error(f'No more people within {age_group} generation')

            sample_size = int(household_row.household_headcount - len(inhabitants))
            households.iat[household_row.household_index, households_prop_unassigned_occupants_idx] = sample_size
            households.iat[household_row.household_index, households_prop_inhabitants_idx] = str(inhabitants)

            # logging.info(f'Population to draw from: {population_to_draw_from[:10]}, sample size {sample_size}')
            if sample_size > 0:
                households_interim[household_row.household_index] = inhabitants

        # 2nd iteration - single age groups
        logging.warning(
            f'There are {len(households[households[entities.h_prop_unassigned_occupants] > 0].index)} empty households')
        logging.warning(
            f'There are {len(households_interim.keys())} empty households vol 2')
        assert len(households[households[entities.h_prop_unassigned_occupants] == -1].index) == 0

        def narrow_age_group(df, young, middle, elderly):
            return df[(df.young == young) & (df.middle == middle) & (df.elderly == elderly)]

        def process_single_age_group(df_h, df_p, df_interim, age_group, young, middle, elderly):
            current_households = df_h.loc[df_interim.keys()]
            df = narrow_age_group(current_households, young, middle, elderly)
            for i, row in tqdm(df.iterrows(), desc=f'Lodging {age_group} population'):
                size = int(row[entities.h_prop_unassigned_occupants])
                new_inhabitants = [_homeless_indices[age_group].pop() for _ in
                                   range(min(size, len(_homeless_indices[age_group])))]
                if len(new_inhabitants) == 0:
                    logging.warning(f'No more people within {age_group} group')
                    return
                df_h.iat[row.household_index, households_prop_inhabitants_idx] = str(
                    df_interim[row.household_index] + new_inhabitants)
                if len(new_inhabitants) == size:
                    df_interim.pop(row.household_index)
                else:
                    df_interim[row.household_index] = size - len(new_inhabitants)
                df_h.iat[row.household_index, households_prop_unassigned_occupants_idx] -= len(new_inhabitants)
                for new_inhabitant in new_inhabitants:
                    df_p.iat[new_inhabitant, population_prop_household_idx] = row.household_index

        process_single_age_group(households, population, households_interim, entities.AgeGroup.young.name, 1, 0, 0)
        process_single_age_group(households, population, households_interim, entities.AgeGroup.middle.name, 0, 1, 0)
        process_single_age_group(households, population, households_interim, entities.AgeGroup.elderly.name, 0, 0, 1)

        def process_multiple_age_groups(df_h, df_p, df_interim, age_groups, young, middle, elderly):
            current_households = df_h.loc[df_interim.keys()]
            homeless_indices = df_p[(df_p[entities.prop_household] == entities.HOUSEHOLD_NOT_ASSIGNED)
                                    & (df_p['generation'].isin(age_groups))].index.tolist()
            random.shuffle(homeless_indices)
            _admissible_households = narrow_age_group(current_households, young, middle, elderly)

            for i, row in tqdm(_admissible_households.iterrows(), total=len(_admissible_households.index),
                               desc=f'Lodging {age_groups} population'):
                size = int(row[entities.h_prop_unassigned_occupants])
                new_inhabitants = [homeless_indices.pop() for _ in range(min(size, len(homeless_indices)))]
                if len(new_inhabitants) == 0:
                    logging.error(f'Not enough population to lodge in households for {age_groups}')
                    return
                df_h.iat[row.household_index, households_prop_inhabitants_idx] = str(
                    df_interim[row.household_index] + new_inhabitants)
                if len(new_inhabitants) == size:
                    df_interim.pop(row.household_index)
                else:
                    df_interim[row.household_index] = size - len(new_inhabitants)
                df_h.iat[row.household_index, households_prop_unassigned_occupants_idx] -= len(new_inhabitants)
                for new_inhabitant in new_inhabitants:
                    df_p.iat[new_inhabitant, population_prop_household_idx] = row.household_index

        process_multiple_age_groups(households, population, households_interim,
                                    (entities.AgeGroup.young.name, entities.AgeGroup.middle.name), 1, 1, 0)
        process_multiple_age_groups(households, population, households_interim,
                                    (entities.AgeGroup.middle.name, entities.AgeGroup.elderly.name), 0, 1, 1)
        process_multiple_age_groups(households, population, households_interim,
                                    (entities.AgeGroup.young.name, entities.AgeGroup.elderly.name), 1, 0, 1)
        process_multiple_age_groups(households, population, households_interim, [x.name for x in entities.AgeGroup], 1,
                                    1, 1)

        logging.info('Cleaning up the population dataframe')
        population = rename_index(age_range_to_age(drop_obsolete_columns(population, entities.columns)),
                                  entities.prop_idx)
        logging.info('Other features')
        process_other_features(population, other_features)
    finally:
        logging.info('Saving a population to a file... ')
        population.to_csv(str(output_folder / datasets.output_population_csv.file_name))

        logging.info('Saving households to a file... ')
        households.to_csv(str(output_folder / datasets.output_households_full_csv.file_name), index=False)

        households = drop_obsolete_columns(households, entities.household_columns)
        if gaf is not None:
            gaf = drop_obsolete_columns(gaf.reset_index(), entities.household_columns)
            households = pd.concat((households, gaf))

        households.to_csv(str(output_folder / datasets.output_households_csv.file_name), index=False)

    return population, households


def generate(data_folder: Path, simulations_folder: Path = None,
             other_features: List[Tuple[Feature, FeatureParams]] = []) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a population given the folder with data and the size of this population.
    :param data_folder: folder with data
    :param population_size: size of a population to generate; default is the size of the population of Wrocław
    :param simulations_folder: the path to a folder where population and households for this simulation are to be saved.
    If the folder already exists and contains households.xlsx then households are read from the file. If the folder
    already exists and contains population.xslx file then a population is read from the file.
    :param other_features: map of other features generatorators
    :return: a pandas dataframe with a population generated from the data in data_folder
    """
    # simulations folder
    if simulations_folder is None:
        simulations_folder = project_dir / 'data' / 'simulations' / datetime.now().strftime('%Y%m%d_%H%M')
    if not simulations_folder.is_dir():
        simulations_folder.mkdir()

    return generate_population(data_folder, simulations_folder, other_features)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    city = 'DW'
    data_folder = project_dir / 'data' / 'processed' / 'poland' / city

    other = [(SocialCompetence(), SocialCompetenceParams()),
             (Employment(), EmploymentParams(data_folder))]

    # or to generate a new dataset
    generate(data_folder, other_features=other)
