import pandas as pd
from unittest import TestCase
from src.features import SocialCompetence, SocialCompetenceParams
from src.data.entities import prop_social_competence


class TestSocialCompetence(TestCase):

    def test_should_generate_social_competence_vector_between_0_and_1(self):
        size = 10
        result = SocialCompetence().generate(size, SocialCompetenceParams(),
                                             pd.DataFrame(index=list(range(size)), columns=['whatever']))
        self.assertEqual(size, result.shape[0])
        self.assertEqual(2, result.shape[1])  # whatever and social competence
        self.assertIn(prop_social_competence, result.columns.tolist())
        for idx, x in result[prop_social_competence].iteritems():
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, 1)
