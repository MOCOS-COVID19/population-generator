from contextlib import closing
from pathlib import Path
from typing import Tuple, List, Optional, Iterator

import mocos_helper
import pandas as pd
from xlrd import XLRDError

from src.data.datasets import *
from src.data.entities import BasicNode, GENDERS, Gender, prop_social_competence, prop_ishealthcare
from src.features import SocialCompetence, SocialCompetenceParams, IsHealthCareParams, IsHealthCare
from src.generation.population_generator import PopulationGenerator
from src.generation.population_generator_common import prepare_simulations_folder, get_age_gender_df


def transform(subpopulation_df: pd.DataFrame) -> Tuple[List[Tuple[int, Gender]], List[float]]:
    ret = []
    probs = []

    for row in subpopulation_df.to_dict(orient='records'):
        fem_prob = row['female_probability']
        male_prob = 1.0 - fem_prob
        fem_prob *= row['total_probability']
        male_prob *= row['total_probability']
        ret.append((row['Age'], GENDERS[0]))
        probs.append(fem_prob)
        ret.append((row['Age'], GENDERS[1]))
        probs.append(male_prob)
    return ret, probs


def presample_subpop(subpopulation_df: pd.DataFrame, count: int) -> List[Tuple[int, Gender]]:
    V, probs = transform(subpopulation_df)
    idxes = mocos_helper.sample_with_replacement_shuffled(probs, count)
    return [V[idx] for idx in idxes]


class PolishPopulationGenerator(PopulationGenerator):
    """Generator class for Polish population. The generation process is split into voivodships, since there is
    separate data on age/gender as well as on number of households within each voivodship.
    Age and gender are the latest known. The household data is the prognosis done by GUS in 2016 for the year 2020. """

    def __init__(self, data_folder: Path, voivodship: Path) -> None:
        """Given a path to a voivodship folder, the initializer reads:
         * age/gender dataframe and splits it into children (<18 y.o.) and adults (18+)
         * households dataframe (that contains joint distribution of number of children x number of adults in a
         household. """
        super().__init__(data_folder)
        self.voivodship = voivodship
        self.age_gender_df = get_age_gender_df(self.data_folder, self.voivodship.name)
        self.children_df = self.age_gender_df[self.age_gender_df['Age'] < 18].reset_index(drop=True)
        self.children_df['total_probability'] = self.children_df['Total'] / self.children_df['Total'].sum()
        self.adults_df = self.age_gender_df[self.age_gender_df['Age'] >= 18].reset_index(drop=True)
        self.adults_df['total_probability'] = self.adults_df['Total'] / self.adults_df['Total'].sum()
        self.households_headcount_ac_df = self._preprocess_household_headcount_ac()
        presampled_households_idxes = mocos_helper.sample_with_replacement_shuffled(
            list(self.households_headcount_ac_df.probability),
            int(self.number_of_households))

        self.presampled_households = self.households_headcount_ac_df.iloc[list(presampled_households_idxes)]
        children_needed = self.presampled_households.children.sum()
        adults_needed = self.presampled_households.adults.sum()
        self.presampled_children = presample_subpop(self.children_df, int(children_needed)).__iter__()
        self.presampled_adults = presample_subpop(self.adults_df, int(adults_needed)).__iter__()
        self.presampled_households = self.presampled_households.to_dict(orient='records').__iter__()

    @property
    def number_of_households(self) -> int:
        return self.households_headcount_ac_df.households.sum()

    def _preprocess_household_headcount_ac(self) -> pd.DataFrame:
        """If the household excel for a voivodship was processed already, the function reads the results from the
        relevant excel file and returns them. Otherwise it reads the part of the GUS-provided excel file that is
        corresponding to year 2020 and formats these data into more convenient format. Finally, saves into a file and
        returns the dataframe. """
        try:
            df2 = pd.read_excel(str(self.voivodship / households_headcount_ac_xlsx.file_name),
                                sheet_name=households_headcount_ac_xlsx.sheet_name)
        except XLRDError:
            # read raw data - skip rows corresponding to previous years, read only this year and skip the total column
            df = pd.read_excel(str(self.voivodship / households_headcount_ac_xlsx_raw.file_name),
                               sheet_name=households_headcount_ac_xlsx_raw.sheet_name, header=None,
                               skiprows=29, nrows=6, usecols=[1, 3, 4, 5, 6, 7, 8],
                               names=['children', 0, 1, 2, 3, 4, 5], index_col=0)

            # adults=0, children=0 is NaN, fix that and convert to int
            df = df.fillna(0)
            df = df.astype(int)

            # make all indices int
            as_list = df.index.tolist()
            idx = as_list.index('5+')
            as_list[idx] = 5
            df.index = as_list
            df.index.name = 'children'

            # melt the dataframe into three columns: children, adults, households (number of occurrences)
            df2 = pd.melt(df.reset_index(), id_vars=['children'], var_name='adults', value_name='households')
            df2 = df2.astype(int)
            # get the total headcount in a household
            df2['headcount'] = df2.children + df2.adults
            # probability of such a household
            df2['probability'] = df2['households'] / df2['households'].sum()

            # save to excel file
            with closing(pd.ExcelWriter(str(self.voivodship / households_headcount_ac_xlsx.file_name),
                                        engine='openpyxl')) as writer:
                df2.to_excel(writer, sheet_name=households_headcount_ac_xlsx.sheet_name, index=False)
        return df2

    def _draw_a_household(self) -> Tuple[int, int]:
        """Randomly select a household given the probability of occurrence"""
        row = next(self.presampled_households)
        return int(row['children']), int(row['adults'])

    def _draw_from_subpopulation(self, subpopulation: Iterator[Tuple[int, Gender]], headcount: int, household_idx: int,
                                 current_index: int) -> Tuple[List[BasicNode], int]:
        """Randomly draw `headcount` people from `subpopulation` given the probability of age/gender combination within this
        subpopulation and lodge them together in a household given by `household_idx`. """
        nodes = []

        for _ in range(headcount):
            row = next(subpopulation)
            age = row[0]
            gender = row[1]
            nodes.append(BasicNode(current_index, age, gender, household_idx))
            current_index += 1

        return nodes, current_index

    def _draw_children(self, children_count: int, household_idx: int, current_index: int) -> Tuple[
        List[BasicNode], int]:
        """Randomly draw `children_count` children and lodge them together in a household given by `household_idx`.
        """
        return self._draw_from_subpopulation(self.presampled_children, children_count, household_idx, current_index)

    def _draw_adults(self, adults_count: int, household_idx: int, current_index: int) -> Tuple[List[BasicNode], int]:
        """Randomly draw `adults_count` adults and lodge them together in a household given by `household_idx`.
        """
        return self._draw_from_subpopulation(self.presampled_adults, adults_count, household_idx, current_index)

    def _prepare_simulation_folder(self, simulations_parent_folder: Optional[Path] = None) -> Path:
        """Within the given `simulations_folder` create a voivodship folder to save population and households data. """
        simulations_folder = simulations_parent_folder / self.voivodship.name
        simulations_folder.mkdir()
        return simulations_folder

    def _draw_household_and_members(self, current_household_idx, current_index) -> Tuple[List[BasicNode], int]:
        children_count, adults_count = self._draw_a_household()
        children, current_index = self._draw_children(children_count, current_household_idx, current_index)
        adults, current_index = self._draw_adults(adults_count, current_household_idx, current_index)
        return children + adults, current_index


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    poland_folder = project_dir / 'data' / 'processed' / 'poland'
    voivodships = [x for x in poland_folder.iterdir() if x.is_dir() and len(x.name) == 1]
    # TODO: consider changing to subprocesses
    next_household_index = 0
    next_person_index = 0
    poland_simulations_folder = prepare_simulations_folder()
    other_features = {prop_social_competence: (SocialCompetence(), SocialCompetenceParams())}
    for i, item in enumerate(voivodships):
        other_features[prop_ishealthcare] = (IsHealthCare(), IsHealthCareParams(item.name))
        next_household_index, next_person_index = PolishPopulationGenerator(poland_folder, item).run(
            next_household_index,
            next_person_index,
            poland_simulations_folder,
            other_features)
