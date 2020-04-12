from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import fire

from src.data import datasets as ds, entities as en
from src.generation import population_generator_common as common


class NSP:
    ID_OSOBA = 'ID_Osoba'
    ID_Mieszkania = 'ID_Mieszkania'
    ID_Gospodarstwa = 'ID_Gospodarstwa'
    WOJ = 'WOJ'
    POW = 'POW'
    PLEC = 'Plec'
    WIEK = 'Wiek'
    STATUS_AE_ILO = 'Status_AE_ILO'
    ZP_AE_ST_ZATR_GL_ILO = 'ZP_AE_ST_ZATR_GL_ILO'
    PKD_GLOWNAP_KOD_ILO = 'PKD_GLownaP_kod_ILO'
    ZAWOD_WYKON_GLOWNAP_KOD_ILO = 'Zawod_Wykon_GlownaP_kod_ILO'

    datatypes = {ID_OSOBA: int, ID_Mieszkania: int, ID_Gospodarstwa: int, WOJ: str, POW: str, PLEC: int, WIEK: int,
                 STATUS_AE_ILO: float, ZAWOD_WYKON_GLOWNAP_KOD_ILO: str}


class NSPTransformer:
    households_columns = [NSP.ID_OSOBA, NSP.ID_Gospodarstwa, NSP.POW, NSP.PLEC, NSP.WIEK, NSP.STATUS_AE_ILO,
                          NSP.ZAWOD_WYKON_GLOWNAP_KOD_ILO]
    apartment_columns = [NSP.ID_OSOBA, NSP.ID_Mieszkania, NSP.POW, NSP.PLEC, NSP.WIEK, NSP.STATUS_AE_ILO,
                         NSP.ZAWOD_WYKON_GLOWNAP_KOD_ILO]

    column_mapping = {NSP.ID_OSOBA: en.prop_idx, NSP.ID_Gospodarstwa: en.prop_household,
                      NSP.ID_Mieszkania: en.prop_household,
                      NSP.POW: en.prop_powiat,
                      NSP.PLEC: en.prop_gender, NSP.WIEK: en.prop_age,
                      NSP.STATUS_AE_ILO: en.prop_employment_status,
                      NSP.ZAWOD_WYKON_GLOWNAP_KOD_ILO: en.prop_profession}

    employment_mapping = {1.0: en.EmploymentStatus.EMPLOYED.value, 2.0: en.EmploymentStatus.NOT_EMPLOYED.value,
                          3.0: en.EmploymentStatus.NOT_EMPLOYED.value, 9.0: en.EmploymentStatus.NOT_SET.value,
                          np.nan: en.EmploymentStatus.NOT_EMPLOYED.value}

    def __init__(self, input_folder: Path, output_folder_households: Path, output_folder_apartments: Path) -> None:
        self.input_folder = input_folder
        self.output_folder_households = output_folder_households
        self.output_folder_apartments = output_folder_apartments
        self._powiat_subregion_mapping = None

    @property
    def powiat_subregion_mapping(self):
        if self._powiat_subregion_mapping is None:
            project_dir = Path(__file__).resolve().parents[2]
            poland_processed_data_dir = project_dir / 'data' / 'processed' / 'poland'
            data = pd.read_excel(str(poland_processed_data_dir / ds.powiats_subregions_mapping_xlsx.file_name),
                                 dtype=str)
            self._powiat_subregion_mapping = {row['teryt_code']: row['powiat_code'] for _, row in data.iterrows()}
        return self._powiat_subregion_mapping

    @staticmethod
    def read_raw_data(file_path: Path) -> pd.DataFrame:
        return pd.read_excel(str(file_path), converters=NSP.datatypes)

    def decorate_raw_data(self, df):
        df[NSP.PLEC] -= 1  # in NSP it is 1(m), 2(f); in our setup 0(m), 1(f)
        df[NSP.STATUS_AE_ILO] = df[NSP.STATUS_AE_ILO].replace(self.employment_mapping)
        df[NSP.ZAWOD_WYKON_GLOWNAP_KOD_ILO] = df[NSP.ZAWOD_WYKON_GLOWNAP_KOD_ILO].fillna(en.PROFESSION_NOT_ASSIGNED)
        df[NSP.POW] = df[NSP.POW].replace(self.powiat_subregion_mapping)
        return df

    def transform_single(self, file_path: Path):
        df = self.decorate_raw_data(self.read_raw_data(file_path))

        voivodship = file_path.stem[-2:]
        self.transform_save_single(df, self.households_columns, self.output_folder_households, voivodship)
        self.transform_save_single(df, self.apartment_columns, self.output_folder_apartments, voivodship)

    def transform_save_single(self, df: pd.DataFrame, columns: List[str], output_folder: Path, voivodship: str):
        output_folder = output_folder / voivodship
        output_folder.mkdir()
        df_population = df[columns].rename(columns=self.column_mapping)
        df_population.to_csv(str(output_folder / ds.output_population_csv.file_name), index=False)
        df_households = df_population.groupby(en.prop_household)[[en.prop_idx]].aggregate(lambda x: list(x))
        df_households.to_csv(str(output_folder / ds.output_households_csv.file_name))

    def transform(self):
        for file in self.input_folder.iterdir():
            if file.is_file() and file.suffix == '.xlsx':
                self.transform_single(file)


def main(input_path_str: str, results_dir: str) -> None:
    input_path = Path(input_path_str).resolve()
    output_path_households = common.prepare_simulations_folder(Path(f'{results_dir}/households'))
    output_path_apartments = common.prepare_simulations_folder(Path(f'{results_dir}/apartments'))
    NSPTransformer(input_path, output_path_households, output_path_apartments).transform()


if __name__ == '__main__':
    fire.Fire(main)
    # python src\generation\nsp2011.py d:\coronavirus\nsp2011\NSP2011\ nsp2011_powiats
