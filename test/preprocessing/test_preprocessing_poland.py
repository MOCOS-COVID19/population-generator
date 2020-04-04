from unittest import TestCase
from pathlib import Path
from src.data.datasets import generations_configuration_xlsx
from src.preprocessing import preprocessing_poland as poland


class TestGenerationConfiguration(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        project_dir = Path(__file__).resolve().parents[2]
        data_folder = project_dir / 'data' / 'processed' / 'poland' / 'WW'
        assert data_folder.is_dir()
        assert (data_folder / generations_configuration_xlsx.file_name).is_file()
        cls.generations_configuration_df = poland.get_generations_configuration_df(
            data_folder, generations_configuration_xlsx)

    def test_find_configurations_for_family_type_1_selfcontained(self):
        # df, headcount, family_type, relationship, house_master):
        df = poland.draw_generation_configuration_for_household(self.generations_configuration_df, None, 1,
                                                            'Bez osób spoza rodziny', '')
        self.assertEqual(7, len(df.index))

    def test_find_configurations_for_family_type_1_related(self):
        # df, headcount, family_type, relationship, house_master):
        df = poland.draw_generation_configuration_for_household(self.generations_configuration_df, None, 1,
                                                            'Z krewnymi w linii prostej starszego pokolenia',
                                                            'inna osoba')
        self.assertEqual(7, len(df.index))

    def test_find_configurations_for_family_type_1_unrelated(self):
        # df, headcount, family_type, relationship, house_master):
        df = poland.draw_generation_configuration_for_household(self.generations_configuration_df, None, 1,
                                                            'Z innymi osobami', 'członek rodziny')
        self.assertEqual(7, len(df.index))

    def test_find_configurations_for_family_type_1_selfcontained(self):
        # df, headcount, family_type, relationship, house_master):
        df = poland.draw_generation_configuration_for_household(self.generations_configuration_df, None, 1,
                                                            'Bez osób spoza rodziny', '')
        self.assertEqual(7, len(df.index))

    def test_find_configurations_for_family_type_2_unrelated(self):
        df = poland.draw_generation_configuration_for_household(self.generations_configuration_df, None, 2,
                                                            'Niespokrewnione w linii prostej', '')
        self.assertEqual(7, len(df.index))

    def test_find_configurations_for_family_type_2_related(self):
        df = poland.draw_generation_configuration_for_household(self.generations_configuration_df, None, 2,
                                                            'Spokrewnione w linii prostej',
                                                            'członek rodziny starszego pokolenia')
        self.assertEqual(7, len(df.index))

    def test_find_configurations_for_family_type_3(self):
        df = poland.draw_generation_configuration_for_household(self.generations_configuration_df, None, 3, '', '')
        self.assertEqual(7, len(df.index))

    def test_find_configurations_for_family_type_0_one_person(self):
        df = poland.draw_generation_configuration_for_household(self.generations_configuration_df, 1, 0, '', '')
        self.assertEqual(3, len(df.index))  # only those with nb_generations <= headcount [here: 1]

    def test_find_configurations_for_family_type_1_many_people(self):
        df = poland.draw_generation_configuration_for_household(self.generations_configuration_df, 3, 0, '', '')
        self.assertEqual(7, len(df.index))
