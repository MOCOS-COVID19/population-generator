from pathlib import Path
from src.data.entities import BasicNode, household_columns
import csv
from typing import List
from src.data import datasets


class Violation:
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f'{self.__class__.__name__}({self.message})'


class IncorrectHeaderViolation(Violation):
    def __init__(self, expected_header, actual_header):
        super().__init__(f'Expected header "{expected_header}" but got "{actual_header}"')


class HomelessPeopleViolation(Violation):
    def __init__(self, homeless):
        super().__init__(f'There are {homeless} homeless people')


class EmptyHouseholdsViolation(Violation):
    def __init__(self, households, places):
        super().__init__(f'There are {households} empty households capable od lodging {places} people')


def verify_population(simulation_dir: Path) -> List[Violation]:
    violations = []
    with (simulation_dir / datasets.output_population_csv.file_name).open('r') as pop_file:
        reader = csv.reader(pop_file)
        homeless = 0
        for idx, row in enumerate(reader):
            if idx == 0:
                if row != BasicNode.output_fields:
                    violations.append(IncorrectHeaderViolation(row, BasicNode.output_fields))
            else:
                homeless += (row[3] == '-1')

        if homeless > 0:
            violations.append(HomelessPeopleViolation(homeless))

    return violations


def verify_households(simulation_dir: Path) -> List[Violation]:
    violations = []
    with (simulation_dir / datasets.output_households_csv.file_name).open('r') as hh_file:
        reader = csv.reader(hh_file)
        empty_households = 0
        for idx, row in enumerate(reader):
            if idx == 0:
                if row != household_columns:
                    violations.append(IncorrectHeaderViolation(row, household_columns))
            else:
                empty_households += (row[1] == '')

        if empty_households > 0:
            violations.append(EmptyHouseholdsViolation(empty_households, -1))

    return violations


def verify(simulation_dir: Path) -> List[Violation]:
    return verify_population(simulation_dir) + verify_households(simulation_dir)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    data_folder = project_dir / 'data' / 'processed' / 'poland' / 'DW'

    sim_dir = project_dir / 'data' / 'simulations' / '20200402_2212'
    print(verify(sim_dir))
