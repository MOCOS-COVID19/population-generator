from abc import abstractmethod, ABC
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.entities import BasicNode, GENDERS
from src.generation.population_generator_common import nodes_to_dataframe, cleanup_population


class PopulationGenerator(ABC):
    """Generator class for Polish population. The generation process is split into voivodships, since there is
    separate data on age/gender as well as on number of households within each voivodship.
    Age and gender are the latest known. The household data is the prognosis done by GUS in 2016 for the year 2020. """
    household_csv_houshold_index_col = 'household_index'
    household_csv_idx_col = 'idx'
    simulation_population_csv = 'population.csv'
    simulation_household_csv = 'household.csv'

    def __init__(self, data_folder: Path) -> None:
        self.data_folder = data_folder

    def _draw_from_subpopulation(self, subpopulation: pd.DataFrame, headcount: int, household_idx: int,
                                 current_index: int) -> Tuple[List[BasicNode], int]:
        """Randomly draw `headcount` people from `subpopulation` given the probability of age/gender combination within this
        subpopulation and lodge them together in a household given by `household_idx`. """
        nodes = []

        total_probability_col = 'total_probability'
        if total_probability_col in subpopulation.columns.tolist():
            total_probability = subpopulation[total_probability_col]
        else:
            total_probability = subpopulation['Total'] / subpopulation['Total'].sum()

        for _ in range(headcount):
            idx = np.random.choice(subpopulation.index.tolist(), p=total_probability)
            row = subpopulation.loc[idx]
            age = row['Age']
            gender = GENDERS[np.random.choice([0, 1], p=[row.female_probability, 1 - row.female_probability])]
            nodes.append(BasicNode(current_index, age, gender, household_idx))
            current_index += 1

        return nodes, current_index

    @abstractmethod
    def _prepare_simulation_folder(self, simulations_parent_folder: Optional[Path] = None) -> Path:
        """Within the given `simulations_folder` create a voivodship folder to save population and households data. """
        raise NotImplementedError()

    @property
    @abstractmethod
    def number_of_households(self) -> int:
        raise NotImplementedError()

    def _save_interim_results(self, simulation_folder, households, nodes, include_header):
        """Saves (appends) households and population to csv files"""
        hdf = pd.DataFrame(data={self.household_csv_houshold_index_col: list(households.keys()),
                                 self.household_csv_idx_col: list(households.values())})
        pdf = cleanup_population(nodes_to_dataframe(nodes))
        if include_header:
            pdf.to_csv(str(simulation_folder / self.simulation_population_csv), index=False)
            hdf.to_csv(str(simulation_folder / self.simulation_household_csv), index=False)
        else:
            pdf.to_csv(str(simulation_folder / self.simulation_population_csv), mode='a', header=False,
                       index=False)
            hdf.to_csv(str(simulation_folder / self.simulation_household_csv), mode='a', header=False,
                       index=False)

    @abstractmethod
    def _draw_household_and_members(self, current_household_idx, current_index) -> Tuple[List[BasicNode], int]:
        raise NotImplementedError()

    def run(self, household_start_index: Optional[int] = 0, population_start_index: Optional[int] = 0,
            simulation_folder: Optional[Path] = None) -> Tuple[int, int]:
        """Main generation function. Given a starting household_index and a starting population_index, as well as
        the path to a folder where simulation results are to be saved, this function generates population and
        households in a voivodship the generator was initialized with. """
        simulation_folder = self._prepare_simulation_folder(simulation_folder)
        household_batch_size = 1000
        nodes: List[BasicNode] = []
        households: Dict[int, List[int]] = {}
        current_index = population_start_index
        for idx in tqdm(range(self.number_of_households)):
            current_household_idx = idx + household_start_index

            people, current_index = self._draw_household_and_members(current_household_idx, current_index)
            nodes.extend(people)
            households[current_household_idx] = [person.idx for person in people]

            # Every household_batch_size households save the generated households and population and clear memory.
            if idx % household_batch_size == 0 and idx != 0:
                self._save_interim_results(simulation_folder, households, nodes, idx == household_batch_size)
                nodes = []
                households = {}

        include_header = self.number_of_households < household_batch_size
        self._save_interim_results(simulation_folder, households, nodes, include_header)
        return self.number_of_households + household_start_index, current_index
