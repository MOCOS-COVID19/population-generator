from pathlib import Path
from unittest import TestCase
import pandas as pd

from src.data.datasets import employment_rate_by_age_csv, job_market_xlsx
from src.data.entities import prop_industrial_section, prop_employment_status, prop_ishealthcare, \
    Gender
from src.features import EmploymentParams, Employment
from src.generation.population_generator_for_cities import age_gender_generation_population_from_files, age_range_to_age


class TestEmployment(TestCase):
    def setUp(self) -> None:
        project_dir = Path(__file__).resolve().parents[2]
        city = 'DW'
        self.data_folder = project_dir / 'data' / 'processed' / 'poland' / city
        self.resources_folder = Path(__file__).resolve().parent / 'resources'
        self.population = age_gender_generation_population_from_files(self.data_folder)
        self.population = age_range_to_age(self.population)

    def test_employment(self):
        population_size = len(self.population.index)
        population = Employment().generate(EmploymentParams(self.data_folder), self.population)
        population_columns = population.columns.tolist()
        self.assertIn(prop_industrial_section, population_columns)
        self.assertIn(prop_employment_status, population_columns)
        self.assertIn(prop_ishealthcare, population_columns)

    def test_split_population_by_age(self):
        people_by_age = Employment()._split_shuffle_population_by_age(self.population, Gender.FEMALE)
        self.assertEqual(3, len(people_by_age))
        expected_eligible_to_work = len(self.population[(self.population.age >= 15) & (self.population.age < 65)
                                                        & (self.population.gender == Gender.FEMALE.value)].index)
        self.assertEqual(expected_eligible_to_work, sum([len(x) for x in people_by_age.values()]))

    def _prepare_gender_by_age(self, young, middle, middle_immobile):
        return {Employment.young_adults_class: list(range(young)),
                Employment.middle_aged_class: list(range(middle)),
                Employment.middle_aged_immobile_class: list(range(middle_immobile))}

    def test_get_job_market_per_age_group(self):
        employment_feature = Employment()

        employment_rate_per_age = pd.read_csv(str(self.resources_folder / employment_rate_by_age_csv.file_name))
        job_market_df = pd.read_excel(str(self.resources_folder / job_market_xlsx.file_name),
                                      sheet_name=job_market_xlsx.sheet_name)
        gender_by_age = self._prepare_gender_by_age(100, 300, 100)
        gender_column = Employment.females_col
        result = employment_feature._get_job_market_per_age_group(employment_rate_per_age, gender_by_age, job_market_df,
                                                                  gender_column)

        class_1 = [3, 6, 9, 2]
        class_2 = [40, 79, 119, 32]
        class_3 = [7, 15, 22, 6]
        expected_result = pd.DataFrame(data={1: class_1, 2: class_2, 3: class_3, 'id': ['A', 'B', 'C', 'D']}) \
            .set_index('id').astype(int)
        pd.testing.assert_frame_equal(expected_result, result)
        print(result)
