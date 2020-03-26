import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from src.features import entities
from src.data import datasets, preprocessing_poland
from src.features.entities import BasicNode
from src.features.population_generator_common import _age_gender_population, prepare_simulations_folder, \
    get_age_gender_df
from src.features.population_generator import PopulationGenerator
from dataclasses import dataclass


@dataclass
class Household:
    household_index: int
    household_headcount: int
    family_type: int
    relationship: Optional[str]
    house_master: Optional[str]
    young: int
    middle: int
    elderly: int


class CityPopulationGenerator(PopulationGenerator):

    def __init__(self, data_folder: Path):
        super().__init__(data_folder)
        # age and gender merge with production age
        self.age_gender_df = get_age_gender_df(self.data_folder)
        population_size = self.age_gender_df.Total.sum()

        self.households_count_df = pd.read_excel(str(self.data_folder / datasets.households_count_xlsx.file_name),
                                                 sheet_name=datasets.households_count_xlsx.sheet_name)
        # resample the number of households to fit the size of a population
        self.households_count_df['total_people'] = self.households_count_df['nb_of_people_in_household'] \
                                                   * self.households_count_df['nb_of_households']
        self.households_count_df['probability'] = self.households_count_df['nb_of_households'] \
                                                  / self.households_count_df['nb_of_households'].sum()
        old_population_size = self.households_count_df['total_people'].sum()
        self.households_count_df['nb_of_households'] *= (population_size / old_population_size)

        # generation configuration
        self.generations_configuration_df = preprocessing_poland.generate_generations_configuration(self.data_folder)

        # households by master
        self.households_masters_df = pd.read_excel(
            str(self.data_folder / datasets.voivodship_cities_households_by_master_xlsx.file_name),
            sheet_name=datasets.voivodship_cities_households_by_master_xlsx.sheet_name)

        # family structure
        self.family_structure_df = preprocessing_poland.prepare_family_structure_from_voivodship(self.data_folder)

    def _prepare_simulation_folder(self, simulations_parent_folder: Optional[Path] = None) -> Path:
        return prepare_simulations_folder()

    @property
    def number_of_households(self) -> int:
        # return self.households_count_df['nb_of_households'].apply(np.ceil).sum()
        return 3

    def _draw_household_and_members(self, current_household_idx, current_index) -> Tuple[List[BasicNode], int]:
        household = self._draw_a_household(current_household_idx)
        house_masters = narrow_housemasters_by_headcount_and_age_group(self.households_masters_df, household)
        house_masters_probability = house_masters['Count'] / house_masters['Count'].sum()
        index = np.random.choice(house_masters.index.tolist(), p=house_masters_probability)
        masters_age = house_masters.loc[index, 'Age']
        masters_gender = entities.gender_from_string(house_masters.loc[index, 'Gender'])
        nodes = [BasicNode(current_index, masters_age, masters_gender, current_household_idx)]

        for i in range(1, household.household_headcount):
            nodes.append(BasicNode(current_index + i))

        return nodes, current_index + household.household_headcount

    def _draw_a_household(self, current_household_idx: int) -> Household:
        """Randomly select a household given the probability of occurrence"""
        # TODO refactor to a single dataframe
        # household headcount
        idx = np.random.choice(self.households_count_df.index.tolist(),
                               p=self.households_count_df['probability'])
        row = self.households_count_df.loc[idx]

        # family structure
        fs_df = self.family_structure_df[self.family_structure_df.household_headcount == row.nb_of_people_in_household]
        family_idx = np.random.choice(fs_df.index.tolist(), p=fs_df.probability_within_headcount)
        family_structure = fs_df.loc[family_idx]

        # generation configuration
        gc_df = preprocessing_poland.draw_generation_configuration_for_household(self.generations_configuration_df,
                                                                                 family_structure.household_headcount,
                                                                                 family_structure.family_type,
                                                                                 family_structure.relationship,
                                                                                 family_structure.house_master)
        gc_idx = np.random.choice(gc_df.index.tolist(), p=gc_df.probability)
        gc_row = gc_df.loc[gc_idx]

        # household
        household = Household(household_index=current_household_idx,
                              household_headcount=int(row.nb_of_people_in_household),
                              family_type=family_structure.family_type,
                              relationship=family_structure.relationship,
                              house_master=family_structure.house_master,
                              young=gc_row.young, middle=gc_row.middle, elderly=gc_row.elderly)
        return household


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


def generate_population(data_folder: Path, output_folder: Path, households: pd.DataFrame):
    population_ready_xlsx = output_folder / datasets.output_population_xlsx.file_name
    if not population_ready_xlsx.is_file():

        # get this age_gender dataframe and sample for each person
        # or ignore population_size and sum up all
        age_gender_df = pd.read_excel(str(data_folder / datasets.age_gender_xlsx.file_name),
                                      sheet_name=datasets.age_gender_xlsx.sheet_name)

        population = _age_gender_population(age_gender_df)
        population[entities.prop_household] = entities.HOUSEHOLD_NOT_ASSIGNED
        production_age_df = pd.read_excel(str(data_folder.parent / datasets.production_age.file_name),
                                          sheet_name=datasets.production_age.sheet_name)
        population = pd.merge(population, production_age_df, on=['age', 'gender'], how='left')

        logging.info('Finding homeless people...')
        # now we need to lodge other people
        homeless = population[population[entities.prop_household] == entities.HOUSEHOLD_NOT_ASSIGNED]
        homeless_indices = {'young': homeless[homeless.generation == 'young'].index.tolist(),
                            'middle': homeless[homeless.generation == 'middle'].index.tolist(),
                            'elderly': homeless[homeless.generation == 'elderly'].index.tolist()}
        # household_index = population[entities.prop_household]

        logging.info('Selecting households with housemasters and headcount greater than 1...')
        households = households[(households.household_headcount > 1)
                                & (households.house_master != entities.HOUSEHOLD_NOT_ASSIGNED)]

        try:
            for idx, household_row in tqdm(households.iterrows(), desc='Lodging population'):
                lodged_headcount = 1
                try:
                    hm_generation = population.iloc[household_row.house_master]['generation']
                except ValueError as e:
                    logging.error(
                        f'ValueError ({str(e)}) for {household_row.household_index} (house_master={household_row.house_master})')
                    continue
                except TypeError as e:
                    logging.error(
                        f'TypeError ({str(e)}) for {household_row.household_index} (house_master={household_row.house_master})')
                    continue
                # at least one person from each generation

                if household_row.young == 1 and hm_generation != 'young':
                    try:
                        homeless_idx = np.random.choice(homeless_indices['young'])
                        population.loc[homeless_idx, entities.prop_household] = household_row.household_index
                        homeless_indices['young'].remove(homeless_idx)
                        lodged_headcount += 1
                    except ValueError:
                        logging.error('No more people within young generation')

                if household_row.middle == 1 and hm_generation != 'middle':
                    try:
                        homeless_idx = np.random.choice(homeless_indices['middle'])
                        population.loc[homeless_idx, entities.prop_household] = household_row.household_index
                        homeless_indices['middle'].remove(homeless_idx)
                        lodged_headcount += 1
                    except ValueError:
                        logging.error('No more people within middle generation')

                if household_row.elderly == 1 and hm_generation != 'elderly':
                    try:
                        homeless_idx = np.random.choice(homeless_indices['elderly'])
                        population.loc[homeless_idx, entities.prop_household] = household_row.household_index
                        homeless_indices['elderly'].remove(homeless_idx)
                        lodged_headcount += 1
                    except ValueError:
                        logging.error('No more people within young generation')

                sample_size = int(household_row.household_headcount - lodged_headcount)
                # logging.info(f'Population to draw from: {population_to_draw_from[:10]}, sample size {sample_size}')
                if sample_size <= 0:
                    continue

                age_groups_in_household = []
                population_to_draw_from = []
                if household_row.young == 1:
                    age_groups_in_household.append('young')
                    population_to_draw_from.extend(homeless_indices['young'])
                if household_row.middle == 1:
                    age_groups_in_household.append('middle')
                    population_to_draw_from.extend(homeless_indices['middle'])
                if household_row.elderly == 1:
                    age_groups_in_household.append('elderly')
                    population_to_draw_from.extend(homeless_indices['elderly'])

                if len(population_to_draw_from) == 0:
                    logging.error(f'No population to select from for {household_row.household_index}')
                    continue
                try:
                    homeless_idx = np.random.choice(population_to_draw_from, replace=False, size=sample_size)
                except ValueError:
                    logging.error(
                        f'Not enough population to lodge {household_row.household_index}. Taking all {len(population_to_draw_from)} available ({sample_size} needed).')
                    homeless_idx = population_to_draw_from
                population.loc[homeless_idx, entities.prop_household] = household_row.household_index
                for index in homeless_idx:
                    for gen in age_groups_in_household:
                        try:
                            homeless_indices[gen].remove(index)
                            break
                        except ValueError:
                            continue
        finally:
            logging.info('Saving a population to a file... ')
            population.to_excel(str(output_folder / datasets.output_population_xlsx.file_name), index=False)

            logging.info('Saving households to a file... ')
            households.to_excel(str(output_folder / datasets.output_households_xlsx.file_name),
                                sheet_name=datasets.output_households_xlsx.sheet_name, index=False)
    else:
        population = pd.read_excel(str(output_folder / datasets.output_population_xlsx.file_name))
    return population


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    warsaw_folder = project_dir / 'data' / 'processed' / 'poland' / 'WW'

    CityPopulationGenerator(warsaw_folder).run()
