# Poland

Data format and processing is specific for a country and the data and their format provided by this country. 

These data are taken from Polish censuses. The subfolders of this folder contain data for various cities / regions in Poland. File structure and data format in all subfolders is assumed to be the same.

## Files:
* age_gender.xlsx - a dataframe with columns: age, gender, male, female
* healthcare_workers_2017.xlsx - Based on [Biuletyn Statystyczny Ministerstwa Zdrowia](https://www.csioz.gov.pl/fileadmin/user_upload/Biuletyny_informacyjny/biuletyn_2018_5c3deab703e35.pdf) 


## Health care workers for cities
For cities there is no data available that would tell the total number of healthcare workers of all types. In GUS we can find the number of doctors / dentists and nurses / midwifes in each powiat. However, laboratory workers and other personnel are missing. 

To calculate the number of healthcare workers in Wrocław:
1. From GUS (year 2018):
    * in Dolny Śląsk: 6124 doctors in total
    * in Wrocław: 3279 doctors
    * proportion of doctors in Wrocław to doctors in Dolny Śląsk 
    ```
    r = 3279 / 6124
    ```
2. Healthcare workers of all types in Dolny Śląsk: 
    ```
    d = 26298 people
    ```
3. Number of inhabitants (31.12.2017) in Wrocław (from GUS):
    ```
    638 586 = 63.8586 * 10 000
    p = 63.8586
    ```
4. Number of healthcare workers per 10000 inhabitants in Wrocław:
    ```
    r * d / p = 220.5 
    ```

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

 