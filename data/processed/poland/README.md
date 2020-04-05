# Poland

Data format and processing is specific for a country and the data and their format provided by this country. 

These data are taken from Polish censuses. The subfolders of this folder contain data for various cities / regions in Poland. File structure and data format in all subfolders is assumed to be the same.

## Files:
* age_gender.xlsx - a dataframe with columns: age, gender, male, female
* generations.xlsx - mapping of age to a generation: young, middle[-aged], elderly
* powiats.xlsx - list of powiats in Poland
* powiats_subregion_capital_mapping.xlsx - mapping of each powiat to its subregion (a group of powiats based on NUTS-3 classification) and the capital powiat(s)(in 6 cases there are two) of this subregion
* production_age.xlsx - mapping of age and gender into a production age (preproduction, production, postproduction)
* subregions_capitals.xlsx - subregions with their capital powiats
* healthcare_workers_2017.xlsx - Based on [Biuletyn Statystyczny Ministerstwa Zdrowia](https://www.csioz.gov.pl/fileadmin/user_upload/Biuletyny_informacyjny/biuletyn_2018_5c3deab703e35.pdf)

## Naming convention

The naming convention follows the convention utilized in registration plates. 
* first letter - voivodship (required)
* second letter - powiat (optional, if applicable)

### Voivodships
* B – województwo podlaskie
* C – województwo kujawsko-pomorskie
* D – województwo dolnośląskie
* E – województwo łódzkie
* F – województwo lubuskie
* G – województwo pomorskie
* K – województwo małopolskie
* L – województwo lubelskie
* N – województwo warmińsko-mazurskie
* O – województwo opolskie
* P – województwo wielkopolskie
* R – województwo podkarpackie
* S – województwo śląskie
* T – województwo świętokrzyskie
* W – województwo mazowieckie
* Z – województwo zachodniopomorskie

 