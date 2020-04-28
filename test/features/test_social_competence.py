from unittest import TestCase
from src.features import SocialCompetence


class TestSocialCompetence(TestCase):

    def test_should_generate_social_competence_vector_between_0_and_1(self):
        size = 10
        result = SocialCompetence.generate(size)
        self.assertEqual(size, result.shape[0])
        self.assertEqual(1, result.shape[1])
        for x in result:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, 1)
