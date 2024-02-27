from enum import Enum
from typing import Union
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer

SupportedScaler = Union[StandardScaler,MinMaxScaler]
SupportedEncoder = Union[OrdinalEncoder,OneHotEncoder]

class ScalerType(Enum):
    STANDARD = 'Padrão (Z-Score)', lambda: StandardScaler(), 
    MINMAX = 'Normalização Min-Max', lambda: MinMaxScaler()

    def __init__(self, description:str, builder:callable):
       self.description = description
       self.builder = builder

    @classmethod
    def values(self):
        return [x.description for x in ScalerType]

    @classmethod
    def get(self, description:str):
        result =  [x for x in ScalerType if x.description == description]
        return None if len(result) == 0 else result[0]
    
    def build(self) -> SupportedScaler:
        return self.builder()

    def __str__(self) -> str:
        return self.description

class EncoderType(Enum):
    LABEL = 'Ordinal Encoder', lambda: OrdinalEncoder(),
    ONE_HOT = 'One-Hot Encoder', lambda: OneHotEncoder()

    def __init__(self, description:str, builder:callable):
       self.description = description
       self.builder = builder

    @classmethod
    def values(self):
        return [x.description for x in EncoderType]

    @classmethod
    def get(self, description:str):
        result =  [x for x in EncoderType if x.description == description]
        return None if len(result) == 0 else result[0]

    def build(self) -> SupportedEncoder:
        return self.builder()

    def __str__(self) -> str:
        return self.description

class Transformer:
    def __init__(self, df:pd.DataFrame, encode:list[str], scale:list[str], encoderType:EncoderType, scalerType:ScalerType):
        encoders = [(encoderType.build(), [col]) for col in encode]
        scalers = [(scalerType.build(), [col]) for col in scale]
        transformers = encoders + scalers
        transformer = make_column_transformer(
            *transformers,
            remainder='passthrough'
        )
        self.transformer = transformer.fit(df)
    
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.transformer.transform(df),
                           columns=self.transformer.get_feature_names_out())
    
    #TODO conferir se precisa mudar o transformer para o LabelEncoder
    # def __label_transform(self, df:pd.DataFrame):
    #     for c in df.columns:
    #         df[c] = LabelEncoder().fit_transform(df[c].to_numpy())
    #     return df
