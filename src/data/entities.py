"""This script gives a representation of an individual within a population. There are two classes, BaseNode and Node
with basic data of an individual and all data, respectively."""
import numpy as np
from enum import Enum
from typing import Optional, List

prop_idx = 'idx'
prop_age = 'age'
prop_gender = 'gender'
prop_employment_status = 'employment_status'
prop_social_competence = 'social_competence'
prop_public_transport_usage = 'public_transport_usage'
prop_public_transport_duration = 'public_transport_duration'
prop_household = 'household_index'
prop_profession = 'profession'

# auxiliary
prop_age_generation = 'age_generation'

columns = [prop_idx, prop_age, prop_gender, prop_household, prop_employment_status, prop_social_competence,
           prop_public_transport_usage, prop_public_transport_duration, prop_profession]
data_types = {prop_idx: np.uint64,
              prop_age: np.int8,
              prop_gender: np.int8,
              prop_employment_status: np.int8,
              prop_social_competence: np.float,
              prop_public_transport_usage: np.int8,
              prop_public_transport_duration: np.float,
              prop_household: np.uint64,
              prop_profession: np.uint64}

h_prop_household_index = 'household_index'
h_prop_inhabitants = 'idx'
h_prop_house_master_index = 'house_master_index'
h_prop_household_headcount = 'household_headcount'
h_prop_young = 'young'
h_prop_middle = 'middle'
h_prop_elderly = 'elderly'
h_prop_unassigned_occupants = 'unassigned_occupants'
household_columns = [h_prop_household_index, h_prop_inhabitants]


class AgeGroup(Enum):
    young = 0
    middle = 1
    elderly = 2


class Gender(Enum):
    NOT_SET = -1
    MALE = 0
    FEMALE = 1


GENDERS = [Gender.FEMALE, Gender.MALE]


def gender_from_string(string):
    if string == 'M':
        return Gender.MALE
    elif string == 'F':
        return Gender.FEMALE
    raise ValueError('Unknown gender {}'.format(string))


class EmploymentStatus(Enum):
    NOT_SET = -1
    NOT_EMPLOYED = 0
    EMPLOYED = 1


class EconomicalGroup(Enum):
    PRZEDPRODUKCYJNY = 0
    PRODUKCYJNY_MOBILNY = 1
    PRODUKCYJNY_NIEMOBILNY = 2
    POPRODUKCYJNY = 3


HOUSEHOLD_NOT_ASSIGNED = -1
PROFESSION_NOT_ASSIGNED = -1
SOCIAL_COMPETENCE_NOT_ASSIGNED = -1
AGE_NOT_SET = -1
PUBLIC_TRANSPORT_USAGE_NOT_SET = -1
PUBLIC_TRANSPORT_DURATION_NOT_SET = -1


class BasicNodeMeta(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls._output_fields = [prop_idx, prop_age, prop_gender, prop_household]

    @property
    def output_fields(cls):
        return cls._output_fields


class BasicNode(dict, metaclass=BasicNodeMeta):

    def __init__(self, idx: int, age: int = AGE_NOT_SET,
                 gender: Gender = Gender.NOT_SET,
                 household: int = HOUSEHOLD_NOT_ASSIGNED,
                 age_generation: Optional[str] = '') -> None:
        """
        Creates a node representing a person.
        :param age: (optional) age of the node, defaults to AGE_NOT_SET
        :param gender: (optional) gender of the node, defaults to Gender.NOT_SET
        :param household: (optional) household index of the node, defaults to HOUSEHOLD_NOT_ASSIGNED
        :return: None
        """
        super().__init__()
        self[prop_idx] = idx
        self[prop_age] = age
        self[prop_gender] = gender.value
        self[prop_household] = household
        self[prop_age_generation] = age_generation

    @property
    def idx(self) -> int:
        return self[prop_idx]

    @property
    def age(self) -> int:
        return self[prop_age]

    @age.setter
    def age(self, age: int) -> None:
        self[prop_age] = age

    @property
    def gender(self) -> int:
        return self[prop_gender]

    @gender.setter
    def gender(self, gender: Gender) -> None:
        self[prop_gender] = gender.value

    @property
    def household(self) -> int:
        return self[prop_household]

    @household.setter
    def household(self, household: int) -> None:
        self[prop_household] = household

    @property
    def economical_group(self) -> EconomicalGroup:
        if self.age < 18:
            return EconomicalGroup.PRZEDPRODUKCYJNY
        if self.age < 45:
            return EconomicalGroup.PRODUKCYJNY_MOBILNY
        if self.gender == Gender.FEMALE.value and self.age < 60:
            return EconomicalGroup.PRODUKCYJNY_NIEMOBILNY
        if self.gender == Gender.MALE.value and self.age < 65:
            return EconomicalGroup.PRODUKCYJNY_NIEMOBILNY
        return EconomicalGroup.POPRODUKCYJNY

    @property
    def age_generation(self) -> str:
        return self[prop_age_generation]

    @property
    def young(self) -> bool:
        return self.age_generation == 'young'

    @property
    def middle_aged(self) -> bool:
        return self.age_generation == 'middle'

    @property
    def elderly(self) -> bool:
        return self.age_generation == 'elderly'

#    @classmethod
#    def output_fields(cls) -> List[str]:
#        return cls._output_fields


class Node(BasicNode):
    _output_fields = [prop_idx, prop_age, prop_gender, prop_household, prop_employment_status, prop_social_competence,
                      prop_public_transport_usage, prop_public_transport_duration, prop_profession]

    def __init__(self, age: int = AGE_NOT_SET,
                 gender: Gender = Gender.NOT_SET,
                 employment_status: EmploymentStatus = EmploymentStatus.NOT_SET,
                 social_competence: float = SOCIAL_COMPETENCE_NOT_ASSIGNED,
                 public_transport_usage: float = PUBLIC_TRANSPORT_USAGE_NOT_SET,
                 public_transport_duration: float = PUBLIC_TRANSPORT_DURATION_NOT_SET,
                 household: int = HOUSEHOLD_NOT_ASSIGNED,
                 profession: int = PROFESSION_NOT_ASSIGNED,
                 age_generation: Optional[str] = '') -> None:
        """
            Creates a node representing a person.
            :param age: (optional) age of the node, defaults to AGE_NOT_SET
            :param gender: (optional) gender of the node, defaults to Gender.NOT_SET
            :param employment_status: (optional) employement status of the node, defaults to EmploymentStatus.NOT_SET
            :param social_competence: (optional) social competence of the node, defaults to SOCIAL_COMPETENCE_NOT_ASSIGNED
            :param public_transport_usage: (optional) public transport usage of the node. in essence binary, but can also be used as
            frequency, defaults to PUBLIC_TRANSPORT_USAGE_NOT_SET (#TODO: to be decided)
            :param public_transport_duration: (optional) mean duration per day spent in public transport (#TODO: to be decided about
            mean vs other aggregate function), defaults to PUBLIC_TRANSPORT_DURATION_NOT_SET
            :param household: (optional) household index of the node, defaults to HOUSEHOLD_NOT_ASSIGNED
            :param profession: (optional) profession index of the node, defaults to PROFESSION_NOT_ASSIGNED
            :param age_generation: (optional) age_generation of an individual
            :return: None
        """
        super().__init__(0, age, gender, household, age_generation)
        self[prop_employment_status] = employment_status.value
        self[prop_social_competence] = social_competence
        self[prop_public_transport_usage] = public_transport_usage
        self[prop_public_transport_duration] = public_transport_duration
        self[prop_profession] = profession

    @property
    def employment_status(self) -> int:
        return self[prop_employment_status]

    @employment_status.setter
    def employment_status(self, employment_status: EmploymentStatus) -> None:
        self[prop_employment_status] = employment_status.value

    @property
    def social_competence(self) -> float:
        return self[prop_social_competence]

    @social_competence.setter
    def social_competence(self, social_competence: float) -> None:
        self[prop_social_competence] = social_competence

    @property
    def public_transport_usage(self) -> float:
        return self[prop_public_transport_usage]

    @public_transport_usage.setter
    def public_transport_usage(self, public_transport_usage: float) -> None:
        self[prop_public_transport_usage] = public_transport_usage

    @property
    def public_transport_duration(self) -> float:
        return self[prop_public_transport_duration]

    @public_transport_duration.setter
    def public_transport_duration(self, public_transport_duration: float) -> None:
        self[prop_public_transport_duration] = public_transport_duration

    @property
    def profession(self) -> int:
        return self[prop_profession]

    @profession.setter
    def profession(self, profession: int) -> None:
        self[prop_profession] = profession

    @classmethod
    def output_fields(cls) -> List[str]:
        return cls._output_fields
