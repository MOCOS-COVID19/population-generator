from typing import Union
from pathlib import Path
import numpy as np
import pandas as pd

from src.features import FeatureParams, Feature
from src.data.datasets import healthcare_workers_xlsx
from src.data.entities import prop_gender, prop_ishealthcare, Gender, prop_age, AgeGroup


class IsHealthCareParams(FeatureParams):
    def __init__(self, territorial_unit):
        self.territorial_unit = territorial_unit


class IsHealthCare(Feature):

    healthcare_column = 'healthcare_per_10000'

    def __init__(self):
        project_dir = Path(__file__).resolve().parents[2]
        healthcare_workers_file = project_dir / 'data' / 'processed' / 'poland' / healthcare_workers_xlsx.file_name
        self.healthcare_workers = pd.read_excel(healthcare_workers_file, index_col=0)

    def generate(self, population_size: int, params: IsHealthCareParams,
                 population: pd.DataFrame) -> Union[pd.Series, np.ndarray]:
        females = population[(population[prop_gender] == Gender.FEMALE.value)
                             & (population[prop_age] >= 22) & (population[prop_age] < 60)].index.tolist()
        males = population[(population[prop_gender] == Gender.MALE.value)
                             & (population[prop_age] >= 22) & (population[prop_age] < 65)].index.tolist()
        # 1 male : 4 females in healthcare
        hw_count = self.healthcare_workers.loc[params.territorial_unit, self.healthcare_column] * population_size / 10000
        hwf = int(round(0.8 * hw_count))
        hwm = int(round(0.2 * hw_count))
        hwfc = np.random.choice(females, size=hwf, replace=False)
        hwfm = np.random.choice(males, size=hwm, replace=False)
        ishealthcare = pd.Series(index=population.index, data=0)
        ishealthcare.loc[hwfc] = 1
        ishealthcare.loc[hwfm] = 1
        population[prop_ishealthcare] = ishealthcare
        return population



