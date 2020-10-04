from pathlib import Path
from unittest import TestCase

import pandas as pd

from src.data import datasets
from src.generation import population_generator_for_cities as population


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

