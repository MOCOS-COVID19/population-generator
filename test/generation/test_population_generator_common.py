from unittest import TestCase
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

from src.data import entities
from src.generation import population_generator_common as population


class TestNodesToDataFrame(TestCase):

    def test_should_convert_LD_to_DL(self):
        ld = [dict(a=1, b=2), dict(a=2, b=3)]
        dl = population._list_of_dicts_to_dict_of_lists(ld)

        self.assertEqual(2, len(dl.keys()))
        self.assertIn('a', dl.keys())
        self.assertIn('b', dl.keys())
        self.assertEqual([1, 2], dl['a'])
        self.assertEqual([2, 3], dl['b'])

    def test_should_convert_nodes_to_dataframe(self):
        node1 = entities.Node(age=2, gender=entities.Gender.MALE,
                              employment_status=entities.EmploymentStatus.NOT_EMPLOYED,
                              social_competence=0.3,
                              public_transport_duration=0,
                              public_transport_usage=False,
                              household=1, industrial_section=entities.INDUSTRIAL_SECTION_NOT_ASSIGNED,
                              age_generation=entities.AgeGroup.young.value, is_healthcare=0)
        node2 = entities.Node(age=37, gender=entities.Gender.FEMALE,
                              employment_status=entities.EmploymentStatus.EMPLOYED,
                              social_competence=0.7,
                              public_transport_duration=30,
                              public_transport_usage=True,
                              household=1, industrial_section='A',
                              age_generation=entities.AgeGroup.middle.value, is_healthcare=1)
        nodes = [node1, node2]

        df = population.nodes_to_dataframe(nodes)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(entities.columns) + 1, len(df.columns))  # age_generation is not in a final dataframe
        self.assertEqual(2, len(df.index))
        for col in entities.columns:
            self.assertIn(col, df.columns.tolist())

        self.assertNodeAndRowEqual(node1, df.iloc[0])
        self.assertNodeAndRowEqual(node2, df.iloc[1])

    def assertNodeAndRowEqual(self, node, row):
        self.assertEqual(node.age, row[entities.prop_age])
        self.assertEqual(node.gender, row[entities.prop_gender])
        self.assertEqual(node.employment_status, row[entities.prop_employment_status])
        self.assertEqual(node.social_competence, row[entities.prop_social_competence])
        self.assertEqual(node.public_transport_usage, row[entities.prop_public_transport_usage])
        self.assertEqual(node.public_transport_duration, row[entities.prop_public_transport_duration])
        self.assertEqual(node.household, row[entities.prop_household])
        self.assertEqual(node.industrial_section, row[entities.prop_industrial_section])


class TestDistributionRetrieval(TestCase):

    def test_should_raise_error_on_unknown_datatype(self):
        try:
            population.get_distribution(1)
            self.fail('Should have raised ValueError')
        except ValueError as e:
            self.assertTrue(str(e).startswith('Expected the name of a distribution'))

    def test_should_raise_error_on_unknown_distribution(self):
        try:
            population.get_distribution('giberish')
            self.fail('Should have raised an error')
        except Exception as e:
            self.assertEqual('module \'scipy.stats\' has no attribute \'giberish\'', str(e))

    def test_should_return_normal_distribution(self):

        distribution = population.get_distribution('norm')
        self.assertIsInstance(distribution, scipy.stats._continuous_distns.norm_gen)


class TestDrawingFromDistribution(TestCase):

    def test_should_draw_10_samples_from_normal_distribution(self):
        sample = population.sample_from_distribution(10, 'norm', 0, 1)
        self.assertEqual(10, len(sample))
        self.assertIsInstance(sample, np.ndarray)


class TestCleanup(TestCase):

    def test_age_transformation(self):
        df = pd.read_excel(str(Path(__file__).resolve().parents[0] / 'test_cleanup_age_parsing.xlsx'))
        result = population.age_range_to_age(df)

        self.assertEqual(int, result[entities.prop_age].dtype)
        for idx, row in result.iterrows():
            self.assertGreaterEqual(row.age, row.age_expected)
            self.assertLess(row.age, row.age_expected + 5)