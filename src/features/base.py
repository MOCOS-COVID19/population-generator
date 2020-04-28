import numpy as np
import pandas as pd
from typing import Union


class FeatureParams:
    pass


class Feature:
    @staticmethod
    def generate(population_size: int, params: FeatureParams = FeatureParams()) -> Union[pd.Series, np.ndarray]:
        raise NotImplementedError('Each feature should implement this')
