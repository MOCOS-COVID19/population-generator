import numpy as np
from src.features.base import Feature, FeatureParams
from src.generation.population_generator_common import (sample_from_distribution)


class SocialCompetenceParams(FeatureParams):
    def __init__(self, distribution_name='norm', loc=0.5, scale=0.15):
        """
        Parameters for SocialCompetence feature
        :param distribution_name: name of a distribution
        :param loc: parameters of the distribution
        :param scale: parameter of the distribution
        """
        self.distribution_name = distribution_name
        self.loc = loc
        self.scale = scale


class SocialCompetence(Feature):
    @staticmethod
    def generate(population_size: int,
                 params: SocialCompetenceParams = SocialCompetenceParams()) -> np.ndarray:
        """
        After [1] social competence (introversion and extraversion) are modelled according to a normal distribution with
        mean shown by the majority of the population.
        [1]  B.Zawadzki, J.Strelau, P.Szczepaniak, M.Śliwińska: Inwentarz osobowości NEO-FFI Costy i McCrae.
        Warszawa: Pracownia Testów Psychologicznych Polskiego Towarzystwa Psychologicznego, 1997. ISBN 83-85512-89-6.
        :param population_size: size of a sample
        :param params: parameters of this feature
        :return: social competence vector of a population
        """
        x = sample_from_distribution(population_size, params.distribution_name, loc=params.loc,
                                     scale=params.scale)
        return np.clip(x, 0, 1).reshape(-1, 1)
