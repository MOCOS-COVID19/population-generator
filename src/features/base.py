import numpy as np
import pandas as pd
from typing import Union


class FeatureParams:
    pass


class Feature:
    def generate(self, population_size: int, params: FeatureParams, population: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError('Each feature should implement this')
