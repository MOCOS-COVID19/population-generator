from src.features import healthcare
from unittest import TestCase
import pandas as pd
from src.data.entities import prop_age, prop_gender, prop_ishealthcare


class TestHealthcare(TestCase):
    def test_healthcare(self):
        age = list(range(0, 85))
        gender = [0] * len(age) + [1] * len(age)
        age = age * 2000
        gender = gender * 1000

        population = pd.DataFrame(data={prop_gender: gender, prop_age: age})
        population_size = len(population.index)
        featured_population = healthcare.IsHealthCare() \
            .generate(healthcare.IsHealthCareParams('D'), population)
        self.assertEqual(population_size, len(featured_population.index))
        self.assertIn(prop_ishealthcare, featured_population.columns.tolist())
        # too young
        youngsters = featured_population.loc[featured_population[prop_age] < 22, prop_ishealthcare]
        self.assertEqual(0, youngsters.sum())

        # too old
        elderly_females = featured_population.loc[
            (featured_population[prop_age] >= 60) & (featured_population[prop_gender] == 1), prop_ishealthcare]
        self.assertEqual(0, elderly_females.sum())
        elderly_males = featured_population.loc[
            (featured_population[prop_age] >= 65) & (featured_population[prop_gender] == 0), prop_ishealthcare]
        self.assertEqual(0, elderly_males.sum())

        middle_count = population_size - len(youngsters.index) - len(elderly_females.index) - len(elderly_males.index)
        self.assertLessEqual(featured_population[prop_ishealthcare].sum(), middle_count)
        self.assertGreaterEqual(featured_population[prop_ishealthcare].sum(), 0)

        print(featured_population[prop_ishealthcare].sum())
