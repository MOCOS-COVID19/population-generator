from unittest import TestCase
from src.features import population_generator_for_cities as population
from src.features import entities
from src.data import datasets
import scipy.stats
import numpy as np
import pandas as pd
from pathlib import Path



class TestSocialCompatenece(TestCase):

    def test_should_generate_social_competence_vector_between_0_and_1(self):
        size = 10
        result = population.generate_social_competence(size)
        self.assertEqual(size, result.shape[0])
        self.assertEqual(1, result.shape[1])
        for x in result:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, 1)


class TestNarrowHousemastersByAgeAndFamilyStructure(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        test_file = Path(__file__).resolve().parents[0] / 'households_combinations.xlsx'
        project_dir = Path(__file__).resolve().parents[2]
        data_folder = project_dir / 'data' / 'processed' / 'poland' / 'DW'
        assert (data_folder / datasets.voivodship_cities_households_by_master_xlsx.file_name).is_file()
        cls.household_by_master = pd.read_excel(
            str(data_folder / datasets.voivodship_cities_households_by_master_xlsx.file_name),
            sheet_name=datasets.households_by_master_xlsx.sheet_name)

        assert test_file.is_file()
        cls.households_rows = pd.read_excel(str(test_file))

    def test_stuff(self):
        for i, row in self.households_rows.iterrows():
            masters = population.narrow_housemasters_by_headcount_and_age_group(self.household_by_master, row)
            self.assertGreaterEqual(len(masters.index), 0)


class TestEmployment(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        project_dir = Path(__file__).resolve().parents[2]
        cls.data_folder = project_dir / 'data' / 'processed' / 'poland' / 'DW'
        age_gender_df = pd.read_excel(str(cls.data_folder / datasets.age_gender_xlsx.file_name),
                                      sheet_name=datasets.age_gender_xlsx.sheet_name)
        cls.pop = population.age_gender_population(age_gender_df)
        cls.population_size = len(cls.pop.index)

    def test_generate_employment(self):
        employment = population.generate_employment(self.data_folder, self.pop)
        self.assertEqual(len(employment.index), self.population_size)


class TestPublicTransport(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        project_dir = Path(__file__).resolve().parents[2]
        cls.data_folder = project_dir / 'data' / 'processed' / 'poland' / 'DW'
        age_gender_df = pd.read_excel(str(cls.data_folder / datasets.age_gender_xlsx.file_name),
                                      sheet_name=datasets.age_gender_xlsx.sheet_name)
        cls.pop = population.age_gender_population(age_gender_df)
        cls.population_size = len(cls.pop.index)

    def test_generate_public_transport(self):
        self.pop[entities.prop_public_transport_usage] = population.generate_public_transport_usage(self.population_size)
        duration = population.generate_public_transport_duration(self.pop[entities.prop_public_transport_usage])
        self.assertEqual(len(duration.index), self.population_size)



