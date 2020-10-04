import random
from typing import List, Dict

import pandas as pd
import numpy as np

from src.data import entities as en
from src.data.datasets import job_market_xlsx, employment_rate_by_age_csv, group_accommodation_facilities_personnel_csv
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
    males_col = 'males'
    females_col = 'females'
    gaf_headcount_col = 'headcount'

    def _assign_workplace_genderwise(self, job_market_share, people: Dict[int, List[int]],
                                     employment_status: pd.Series, industrial_section: pd.Series,
                                     is_healthcare: pd.Series):

        for section_id, section_row in job_market_share.iterrows():
            for age_group in self.working_age_classes:
                for _ in range(section_row[age_group]):
                    employee_idx = people[age_group].pop()
                    employment_status.iat[employee_idx] = en.EmploymentStatus.EMPLOYED.value
                    industrial_section.iat[employee_idx] = section_id
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

        try:
            result = pd.cut(population.loc[(population[en.prop_gender] == gender.value)
                                           & (population[en.prop_gaf_type].isna()), en.prop_age],
                            right=False, bins=age_bins, labels=False)
        except KeyError:
            result = pd.cut(population.loc[(population[en.prop_gender] == gender.value), en.prop_age],
                            right=False, bins=age_bins, labels=False)
        people_by_age = {}
        for age_class in self.working_age_classes:
            people_by_age[age_class] = result[result == age_class].index.tolist()
            random.shuffle(people_by_age[age_class])

        return people_by_age

    def _get_job_market_per_age_group(self, employment_rate_per_age_gender, gender_by_age, job_market_df, gender_column):

        # take the number of people in working age
        number_of_people = {age_group: len(people) for age_group, people in gender_by_age.items()}
        # get employment rate in each age group
        # employment_rate = (employment_rate_per_age_gender.loc[employment_rate_per_age_gender.age_class.isin(
        #     self.working_age_classes), gender_column] / 100).to_dict()
        # get the number of employed people within each age group
        # employment_per_age_group = {age_group: empl_rate_in_group * number_of_people[age_group] for
        #                             age_group, empl_rate_in_group in employment_rate.items()}

        employment_per_age_group = {}
        for age_group in self.working_age_classes:
            employment_rate = employment_rate_per_age_gender.loc\
                                                           [(employment_rate_per_age_gender.age_class == age_group),
                                                            gender_column].iloc[0]
            employed_count = employment_rate * number_of_people[age_group] / 100
            employment_per_age_group[age_group] = employed_count

        # calculate the total number of employed
        total_employed = sum(employment_per_age_group.values())
        # find the fraction of employed that come from each age group
        job_market_share = {age_group: employed / total_employed for age_group, employed in
                            employment_per_age_group.items()}
        # scale the job market  so that proportions of people working in each sector hold
        scaled_job_market = job_market_df[gender_column] * total_employed / job_market_df[gender_column].sum()
        # distribute the job market among age groups according to the job market share
        job_market_per_age_group = {age_group: scaled_job_market * job_market_share_in_group for
                                    age_group, job_market_share_in_group in job_market_share.items()}
        job_market_per_age_group['id'] = job_market_df['id']
        # round and cast to int
        return pd.DataFrame(data=job_market_per_age_group).set_index('id').round().astype(int)

    def _assign_employees_to_gafs(self, params, gaf_houses, is_healthcare, gaf_employee):
        df = pd.read_csv(str(params.data_folder / group_accommodation_facilities_personnel_csv.file_name))
        healhcare_index = list(is_healthcare[is_healthcare == en.HealthCare.YES.value].index)
        random.shuffle(healhcare_index)
        for idx, gaf in gaf_houses.iterrows():
            # facility_type_id,facility_type_name,employees_per_resident
            employees_per_resident = df.loc[df.facility_type_id == gaf[en.prop_gaf_type], 'employees_per_resident'].iloc[0]
            personnel_count = int(np.max((1, np.round(gaf[self.gaf_headcount_col] * employees_per_resident))))
            for i in range(personnel_count):
                employee_idx = healhcare_index.pop()
                gaf_employee.iat[employee_idx] = gaf[en.prop_household]

    def generate(self, params: EmploymentParams, population: pd.DataFrame) -> pd.DataFrame:
        employment_status = pd.Series(index=population.index, data=en.EmploymentStatus.NOT_EMPLOYED.value)
        industrial_section = pd.Series(index=population.index, data='')
        is_healthcare = pd.Series(index=population.index, data=en.HealthCare.NO.value)
        gaf_employee = pd.Series(index=population.index, data=en.GAF_EMPLOYEE_NOT_ASSIGNED)

        # shuffle people in the population
        females_by_age = self._split_shuffle_population_by_age(population, en.Gender.FEMALE)
        males_by_age = self._split_shuffle_population_by_age(population, en.Gender.MALE)

        # employment rate in age group
        employment_rate_by_age = pd.read_csv(str(params.data_folder / employment_rate_by_age_csv.file_name))
        # columns: age_range, males, females, age_class

        # number of working people in the city
        job_market = pd.read_excel(str(params.data_folder / job_market_xlsx.file_name),
                                   sheet_name=job_market_xlsx.sheet_name)
        # columns: id, company_size, males, females

        females_job_market = self._get_job_market_per_age_group(employment_rate_by_age, females_by_age, job_market,
                                                                self.females_col)
        males_job_market = self._get_job_market_per_age_group(employment_rate_by_age, males_by_age, job_market,
                                                              self.males_col)

        # allocate people to jobs
        self._assign_workplace_genderwise(females_job_market, females_by_age, employment_status, industrial_section,
                                          is_healthcare)
        self._assign_workplace_genderwise(males_job_market, males_by_age, employment_status, industrial_section,
                                          is_healthcare)
        # allocate employees to gaf's
        try:
            gaf_houses = population[~population[en.prop_gaf_type].isna()]\
                .groupby(by=[en.prop_household, en.prop_gaf_type]).size().reset_index()\
                .rename(columns={0: self.gaf_headcount_col})
            self._assign_employees_to_gafs(params, gaf_houses, is_healthcare, gaf_employee)
        except KeyError:
            pass

        population[en.prop_employment_status] = employment_status
        population[en.prop_industrial_section] = industrial_section
        population[en.prop_ishealthcare] = is_healthcare
        population[en.prop_gaf_employee] = gaf_employee
        return population
