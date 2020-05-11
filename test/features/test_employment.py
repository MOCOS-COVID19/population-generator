from pathlib import Path
from unittest import TestCase
import pandas as pd
import numpy as np

from src.data.datasets import employment_rate_by_age_csv
from src.data.entities import prop_industrial_section, prop_company_size, prop_employment_status, prop_ishealthcare, \
    Gender
from src.features import EmploymentParams, Employment
from src.generation.population_generator_for_cities import age_gender_generation_population_from_files, age_range_to_age


class TestEmployment(TestCase):
    def setUp(self) -> None:
        project_dir = Path(__file__).resolve().parents[2]
        city = 'DW'
        self.data_folder = project_dir / 'data' / 'processed' / 'poland' / city
        self.population = age_gender_generation_population_from_files(self.data_folder)
        self.population = age_range_to_age(self.population)

    def test_employment(self):
        population_size = len(self.population.index)
        population = Employment().generate(population_size, EmploymentParams(self.data_folder), self.population)
        population_columns = population.columns.tolist()
        self.assertIn(prop_industrial_section, population_columns)
        self.assertIn(prop_company_size, population_columns)
        self.assertIn(prop_employment_status, population_columns)
        self.assertIn(prop_ishealthcare, population_columns)

    def test_split_population_by_age(self):
        people_by_age = Employment()._split_shuffle_population_by_age(self.population, Gender.FEMALE)
        self.assertEqual(3, len(people_by_age))
        expected_eligible_to_work = len(self.population[(self.population.age >= 15) & (self.population.age < 65)
                                                        & (self.population.gender == Gender.FEMALE.value)].index)
        self.assertEqual(expected_eligible_to_work, sum([len(x) for x in people_by_age.values()]))

    def test_get_employed_per_age_group(self):
        employment_rate_by_age = pd.read_csv(str(self.data_folder / employment_rate_by_age_csv.file_name))
        employment_feature = Employment()
        females_by_age = employment_feature._split_shuffle_population_by_age(self.population, Gender.FEMALE)
        employment_per_age_group = employment_feature._get_employed_per_age_group(employment_rate_by_age,
                                                                                  females_by_age)
        self.assertEqual(3, len(employment_per_age_group))
        for key, value in employment_per_age_group.items():
            employees = len(females_by_age[key])
            self.assertLessEqual(value, employees)
            self.assertGreater(employees, 0)

    def test_find_number_of_employees(self):
        employment_rate_by_age = pd.read_csv(str(self.data_folder / employment_rate_by_age_csv.file_name))
        employment_feature = Employment()
        females_by_age = employment_feature._split_shuffle_population_by_age(self.population, Gender.FEMALE)
        males_by_age = employment_feature._split_shuffle_population_by_age(self.population, Gender.MALE)
        female_employment_rate_by_age = employment_feature._get_employed_per_age_group(employment_rate_by_age,
                                                                                       females_by_age)
        male_employment_rate_by_age = employment_feature._get_employed_per_age_group(employment_rate_by_age,
                                                                                     males_by_age)
        number_of_employees = sum(female_employment_rate_by_age.values()) + sum(male_employment_rate_by_age.values())
        self.assertLessEqual(0, number_of_employees)
        print(number_of_employees)
