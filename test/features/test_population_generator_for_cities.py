from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
import re

from src.data import datasets
from src.features import entities
from src.features import population_generator_for_cities as gen


class TestPopulation(TestCase):
    def setUp(self) -> None:
        self.resources_dir = Path(__file__).resolve().parents[0] / 'resources'
        self.output_dir = Path(__file__).resolve().parents[0] / 'output'
        assert self.resources_dir.is_dir()  # sanity check
        assert self.output_dir.is_dir()

    def tearDown(self) -> None:
        if (self.resources_dir / datasets.output_households_basic_feather.file_name).is_file():
            (self.resources_dir / datasets.output_households_basic_feather.file_name).unlink()
        for file in self.output_dir.iterdir():
            if file.name != '.gitkeep':
                file.unlink()

    def test_age_gender_generation_population_from_files(self):
        population_size = 1780
        population = gen.age_gender_generation_population_from_files(self.resources_dir)
        self.assertEqual(len(population.index), population_size)

        expected_ages = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                         '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30-34', '35-39',
                         '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84',
                         '85 lat i więcej']
        expected_males = [11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 8, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 7, 7, 8, 9,
                          10, 11, 73, 83, 76, 57, 42, 42, 53, 50, 34, 20, 17, 18]
        expected_females = [10, 10, 10, 10, 9, 9, 9, 9, 9, 10, 9, 9, 8, 8, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 8, 10,
                            12, 13, 84, 93, 84, 61, 46, 50, 69, 68, 51, 35, 34, 38]

        for age, m, f in zip(expected_ages, expected_males, expected_females):
            males = len(population[(population.age == age) & (population.gender == entities.Gender.MALE.value)].index)
            females = len(population[(population.age == age) & (population.gender == entities.Gender.FEMALE.value)].index)
            self.assertEqual(males, m, f'Expected {m} got {males} for age {age}')
            self.assertEqual(females, f, f'Expected {f} got {females} for age {age}')

        self.assertEqual(0, len(population[population.generation.isin(('', np.nan, None))]))

    def test_assign_house_master(self):
        population_size = 1780
        households = gen.generate_households(self.resources_dir, self.output_dir, population_size)
        population = gen.age_gender_generation_population_from_files(self.resources_dir)
        population[entities.prop_household] = entities.HOUSEHOLD_NOT_ASSIGNED
        gen.assign_house_masters(households, population)
        self.assertEqual(len(households[households[entities.h_prop_house_master_index] == -1].index), 0)

        households_len = len(households.index)
        masters_len = len(population[population[entities.prop_household] != entities.HOUSEHOLD_NOT_ASSIGNED].index)
        self.assertEqual(households_len, masters_len)




class TestGeneratePopulation(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resources_dir = Path(__file__).resolve().parents[0] / 'resources'
        cls.output_dir = Path(__file__).resolve().parents[0] / 'output'
        cls.population, cls.households = gen.generate_population(cls.resources_dir, cls.output_dir, False)

    @classmethod
    def tearDownClass(cls) -> None:
        if (cls.resources_dir / datasets.output_households_basic_feather.file_name).is_file():
            (cls.resources_dir / datasets.output_households_basic_feather.file_name).unlink()
        for file in cls.output_dir.iterdir():
            if file.name != '.gitkeep':
                file.unlink()

    def test_population_size(self):
        self.assertEqual(1780, len(self.population.index))

    def test_no_homeless_in_population_df(self):
        self.assertEqual(0, len(self.population[self.population[entities.prop_household]
                                                == entities.HOUSEHOLD_NOT_ASSIGNED]))

    def test_column_names_in_population_df(self):
        self.assertEqual(entities.BasicNode.output_fields, self.population.reset_index().columns.tolist())

    def test_column_names_in_population_csv(self):
        df = pd.read_csv(self.output_dir / datasets.output_population_csv.file_name)
        self.assertEqual(entities.BasicNode.output_fields, df.columns.tolist())

    def test_no_homeless_in_population_csv(self):
        df = pd.read_csv(self.output_dir / datasets.output_population_csv.file_name)
        self.assertEqual(0, len(df[df[entities.prop_household] == entities.HOUSEHOLD_NOT_ASSIGNED]))

    def test_correct_household_columns(self):
        self.assertEqual(entities.household_columns, self.households.columns.tolist())

    def test_household_inhabitants_always_stored_as_list(self):
        pattern = re.compile(r'\[\d+(,\s?\d+)*\]')
        for _, val in self.households[entities.h_prop_inhabitants].iteritems():
            self.assertRegex(val, pattern)

    def test_no_double_household_assignment(self):
        indices = set()
        for _, val in self.households[entities.h_prop_inhabitants].iteritems():
            indices.update(int(x) for x in val.strip('[]').split(', '))
        self.assertEqual(len(self.population.index), len(indices))


class TestGenerateHouseholds(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resources_dir = Path(__file__).resolve().parents[0] / 'resources'
        cls.output_dir = Path(__file__).resolve().parents[0] / 'output'
        cls.population_size = 1780
        cls.households = gen.generate_households(cls.resources_dir, cls.output_dir, cls.population_size)

    @classmethod
    def tearDownClass(cls) -> None:
        if (cls.resources_dir / datasets.output_households_basic_feather.file_name).is_file():
            (cls.resources_dir / datasets.output_households_basic_feather.file_name).unlink()
        for file in cls.output_dir.iterdir():
            if file.name != '.gitkeep':
                file.unlink()

    def test_enough_places_for_all_people(self):
        self.assertGreaterEqual(self.households.household_headcount.sum(), self.population_size)

    def test_correct_numbers_of_households(self):
        self.assertEqual(304, len(self.households[self.households.household_headcount == 1].index))
        self.assertEqual(242, len(self.households[self.households.household_headcount == 2].index))

    def test_existence_of_basic_file(self):
        self.assertTrue((self.output_dir / datasets.output_households_basic_feather.file_name).is_file())

    def test_existence_of_interim_file(self):
        self.assertTrue((self.output_dir / datasets.output_households_interim_feather.file_name).is_file())

    def test_no_more_generations_than_headcount(self):
        self.households['sanity_check'] = self.households['young'] + self.households['middle'] \
                                          + self.households['elderly']
        self.assertEqual(0,
                         len(self.households[self.households['sanity_check'] > self.households['household_headcount']]))

