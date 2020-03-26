import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from src.data import datasets, preprocessing_poland
from src.features import entities
from src.features.entities import BasicNode
from src.features.population_generator import PopulationGenerator
from src.features.population_generator_common import (
    prepare_simulations_folder,
    get_age_gender_df_with_generations)


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
        # age and gender
        self.age_gender_df = get_age_gender_df_with_generations(self.data_folder)
        population_size = self.age_gender_df.Total.sum()

        # generations
        self.young_df = self.age_gender_df[self.age_gender_df.generation == 'young'].copy()
        self.young_df['total_probability'] = self.young_df['Total'] / self.young_df['Total'].sum()
        self.middle_df = self.age_gender_df[self.age_gender_df.generation == 'middle'].copy()
        self.middle_df['total_probability'] = self.middle_df['Total'] / self.middle_df['Total'].sum()
        self.elderly_df = self.age_gender_df[self.age_gender_df.generation == 'elderly'].copy()
        self.elderly_df['total_probability'] = self.elderly_df['Total'] / self.elderly_df['Total'].sum()

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
        return int(self.households_count_df['nb_of_households'].apply(np.ceil).sum())

    def _draw_household_and_members(self, current_household_idx, current_index) -> Tuple[List[BasicNode], int]:
        household = self._draw_a_household(current_household_idx)
        house_masters = narrow_housemasters_by_headcount_and_age_group(self.households_masters_df, household)
        house_masters_probability = house_masters['Count'] / house_masters['Count'].sum()
        index = np.random.choice(house_masters.index.tolist(), p=house_masters_probability)
        masters_age = house_masters.loc[index, 'Age']
        masters_gender = entities.gender_from_string(house_masters.loc[index, 'Gender'])
        master_age_generation = entities.AgeGroup(
            int(house_masters.loc[index, 'middle'] + house_masters.loc[index, 'elderly'] * 2)).name
        master = BasicNode(current_index, masters_age, masters_gender, current_household_idx, master_age_generation)
        return self.generate_population(current_index, household, master)

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

    def generate_population(self, current_index: int, household_row: Household, master: BasicNode) -> Tuple[List[BasicNode], int]:

        lodged_headcount = 1

        inhabitants = [master]
        current_index += 1

        if household_row.young == 1 and not master.young:
            person, current_index = self._draw_from_subpopulation(self.young_df, 1, household_row.household_index,
                                                                  current_index)
            inhabitants.append(person[0])
            lodged_headcount += 1

        if household_row.middle == 1 and not master.middle_aged:
            person, current_index = self._draw_from_subpopulation(self.middle_df, 1, household_row.household_index,
                                                                  current_index)
            inhabitants.append(person[0])
            lodged_headcount += 1

        if household_row.elderly == 1 and not master.elderly:
            person, current_index = self._draw_from_subpopulation(self.elderly_df, 1, household_row.household_index,
                                                                  current_index)
            inhabitants.append(person[0])
            lodged_headcount += 1

        sample_size = int(household_row.household_headcount - lodged_headcount)
        # logging.info(f'Population to draw from: {population_to_draw_from[:10]}, sample size {sample_size}')
        if sample_size <= 0:
            return inhabitants, current_index

        age_groups_in_household = []
        if household_row.young == 1:
            age_groups_in_household.append('young')
        if household_row.middle == 1:
            age_groups_in_household.append('middle')
        if household_row.elderly == 1:
            age_groups_in_household.append('elderly')

        other, current_index = self._draw_from_subpopulation(
            self.age_gender_df[self.age_gender_df.generation.isin(age_groups_in_household)],
            household_row.household_index, sample_size,
            current_index)

        return inhabitants + other, current_index


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


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    warsaw_folder = project_dir / 'data' / 'processed' / 'poland' / 'WW'

    print(CityPopulationGenerator(warsaw_folder).number_of_households)
