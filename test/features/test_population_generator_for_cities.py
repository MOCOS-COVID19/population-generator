# this test should generate a small population
from pathlib import Path
from unittest import TestCase
from src.features import population_generator_for_cities as gen
from src.features import population_generator_common as common
from src.features import entities
from src.data import datasets
import pandas as pd


class TestPopulation(TestCase):
    def setUp(self) -> None:
        self.resources_dir = Path(__file__).resolve().parents[0] / 'resources'
        self.output_dir = Path(__file__).resolve().parents[0] / 'output'
        assert self.resources_dir.is_dir()  # sanity check
        assert self.output_dir.is_dir()

    def tearDown(self) -> None:
        if (self.resources_dir / datasets.households_xlsx.file_name).is_file():
            (self.resources_dir / datasets.households_xlsx.file_name).unlink()
        for file in self.output_dir.iterdir():
            if file.name != '.gitkeep':
                file.unlink()

    def test_generate_households(self):
        population_size = 1780
        households = gen.generate_households(self.resources_dir, self.output_dir, population_size)
        self.assertGreaterEqual(households.household_headcount.sum(), population_size)
        self.assertEqual(304, len(households[households.household_headcount == 1].index))
        self.assertEqual(242, len(households[households.household_headcount == 2].index))
        self.assertTrue((self.resources_dir / datasets.households_xlsx.file_name).is_file())

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

    def test_generate_population(self):
        # two step test
        population = gen.generate_population(self.resources_dir, self.output_dir, False)
        self.assertEqual(1780, len(population.index))
        self.assertEqual(0, len(population[population[entities.prop_household] == entities.HOUSEHOLD_NOT_ASSIGNED]))

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
