import random
from typing import List, Dict
import pandas as pd
import numpy as np

from src.data import entities as en
from src.data.datasets import job_market_xlsx, employment_rate_by_age_csv
from src.features import FeatureParams, Feature


class EmploymentParams(FeatureParams):
    def __init__(self, data_folder):
        self.data_folder = data_folder


class Employment(Feature):
    healthcare_section = 'Q'
    young_adults_class = 1  # 15-24
    middle_aged_class = 2  # 25-54
    middle_aged_immobile_class = 3  # 55-64
    working_age_classes = [young_adults_class, middle_aged_class, middle_aged_immobile_class]

    def _process_gender_section(self, total_number: int, section_id: str, section_company_size: int,
                                people: Dict[int, List[int]], employment_rate_by_age: Dict[int, int],
                                employment_status: pd.Series, industrial_section: pd.Series,
                                is_healthcare: pd.Series, company_size: pd.Series):

        for _ in range(total_number):
            employee_idx = people.pop()
            employment_status.iat[employee_idx] = en.EmploymentStatus.EMPLOYED.value
            industrial_section.iat[employee_idx] = section_id
            company_size.iat[employee_idx] = section_company_size
            if section_id == self.healthcare_section:
                is_healthcare.iat[employee_idx] = en.HealthCare.YES.value

    def _split_shuffle_population_by_age(self, population, gender):
        """
        Splits people of a given gender into age groups. Returns a dictionary with a shuffled list of people per
        age group and the number of children aged 0-14.
        :param population: population
        :param gender: gender to look for in a population
        :return: dictionary of shuffled people per age group and the number of children
        """
        age_bins = [0, 15, 25, 55, 65, 100]
        result = pd.cut(population.loc[(population[en.prop_gender] == gender.value), en.prop_age], right=False,
                        bins=age_bins, labels=False)
        people_by_age = {}
        for age_class in self.working_age_classes:
            people_by_age[age_class] = result[result == age_class].index.tolist()
            random.shuffle(people_by_age[age_class])

        return people_by_age

    def _get_employed_per_age_group(self, employment_rate_per_age, gender_by_age):
        """
        Calculate the number of employed people in each age class, including elderly
        :param employment_rate_per_age: employment rate for total population and all age classes within working age
        :param gender_by_age: people of a specific gender grouped into age groups
        :return: dictionary of number of employed people in each age group
        """
        output_employment_rate = {}
        for age_class in [self.young_adults_class, self.middle_aged_class, self.middle_aged_immobile_class]:
            class_size = len(gender_by_age[age_class])
            class_employment_rate = employment_rate_per_age.loc[employment_rate_per_age.age_class == age_class,
                                                                'percentage'] / 100
            employed_in_class = int(np.round(class_employment_rate * class_size))
            output_employment_rate[age_class] = employed_in_class

        return output_employment_rate

    @staticmethod
    def _scale_job_market(job_market_df, gender_column, gender_employment_rate_by_age):
        return (job_market_df[gender_column] / job_market_df[gender_column].sum() \
                * sum(gender_employment_rate_by_age.values())).round(0).astype(int)

    def generate(self, population_size: int, params: EmploymentParams,
                 population: pd.DataFrame) -> pd.DataFrame:

        employment_status = pd.Series(index=population.index, data=en.EmploymentStatus.NOT_EMPLOYED.value)
        industrial_section = pd.Series(index=population.index, data='')
        is_healthcare = pd.Series(index=population.index, data=en.HealthCare.NO.value)
        company_size = pd.Series(index=population.index, data=0)

        # shuffle people in the population
        females_by_age = self._split_shuffle_population_by_age(population, en.Gender.FEMALE)
        males_by_age = self._split_shuffle_population_by_age(population, en.Gender.MALE)

        # employment rate in age group
        employment_rate_by_age = pd.read_csv(str(params.data_folder / employment_rate_by_age_csv.file_name))
        # columns: age_range, percentage, age_class
        female_employment_rate_by_age = self._get_employed_per_age_group(employment_rate_by_age, females_by_age)
        male_employment_rate_by_age = self._get_employed_per_age_group(employment_rate_by_age, males_by_age)

        # number of working people in the city
        job_market = pd.read_excel(str(params.data_folder / job_market_xlsx.file_name),
                                   sheet_name=job_market_xlsx.sheet_name)
        # columns: id, company_size, males, females
        # scale to much the number of employees in the population
        job_market.males = self._scale_job_market(job_market, 'males', male_employment_rate_by_age)
        job_market.females = self._scale_job_market(job_market, 'females', female_employment_rate_by_age)

        # allocate people to jobs
        for section_idx, section in job_market.iterrows():
            self._process_gender_section(section.females, section.id, section.company_size, females_by_age,
                                         employment_rate_by_age,
                                         employment_status, industrial_section, is_healthcare, company_size)
            self._process_gender_section(section.males, section.id, section.company_size, males_by_age,
                                         employment_rate_by_age,
                                         employment_status, industrial_section, is_healthcare, company_size)

        population[en.prop_employment_status] = employment_status
        population[en.prop_industrial_section] = industrial_section
        population[en.prop_ishealthcare] = is_healthcare
        population[en.prop_company_size] = company_size
        return population
