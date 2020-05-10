from dataclasses import dataclass


@dataclass
class XlsxFile:
    """Data structure representing an Excel file (XLSX format) given by the name of the file and the name of a sheet
    to read data from"""
    file_name: str
    sheet_name: str


@dataclass
class CsvFile:
    file_name: str


@dataclass
class FeatherFile:
    file_name: str


age_gender_xlsx = XlsxFile('age_gender.xlsx', 'processed')
families_and_children_xlsx = XlsxFile('families_and_children.xlsx', 'processed')
families_per_household_xlsx = XlsxFile('families_per_household.xlsx', 'Sheet1')
generations_configuration_xlsx = XlsxFile('generations_configuration.xlsx', 'processed')
household_family_structure_xlsx = XlsxFile('household_family_structure.xlsx', 'Sheet1')
household_family_structure_old_xlsx = XlsxFile('household_family_structure_old.xlsx', 'Sheet1')
households_count_xlsx = XlsxFile('households_count.xlsx', 'processed')
households_old_xlsx = XlsxFile('households_old.xlsx', 'Sheet1')

households_by_master_xlsx = XlsxFile('households_by_master.xlsx', 'House_Master')

output_households_basic_feather = FeatherFile('households_basic.feather')
output_households_interim_feather = FeatherFile('households_interim.feather')
output_households_xlsx = XlsxFile('households.xlsx', 'Sheet1')
output_population_xlsx = XlsxFile('population.xlsx', 'Sheet1')

output_households_full_csv = CsvFile('households_full.csv')
output_households_csv = CsvFile('households.csv')
output_population_csv = CsvFile('population.csv')
production_age = XlsxFile('production_age.xlsx', 'Sheet1')

households_headcount_ac_xlsx_raw = XlsxFile('households_headcount_ac.xlsx', 'Tabl3')
households_headcount_ac_xlsx = XlsxFile('households_headcount_ac.xlsx', 'processed')

voivodship_cities_households_by_master_xlsx = XlsxFile('voivodship_cities_households_by_master.xlsx', 'House_Master')
voivodship_cities_generations_configuration_xlsx = XlsxFile('voivodship_cities_generations_configuration.xlsx',
                                                            'processed')
voivodship_cities_household_family_structure_xlsx = XlsxFile('voivodship_cities_household_family_structure.xlsx',
                                                             'processed')

generations_xlsx = XlsxFile('generations.xlsx', 'Sheet1')

healthcare_workers_xlsx = XlsxFile('healthcare_workers.xlsx', 'Sheet1')
job_market_xlsx = XlsxFile('job_market.xlsx', 'processed')
employment_rate_by_age_csv = CsvFile('employment_rate_by_age.csv')
social_care_houses_csv = CsvFile('social_care_houses.csv')
