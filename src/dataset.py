import pandas as pd

from pipelines.decision_tree_pipeline import *

df = pd.read_parquet('../resources/processed/train.parquet')
df_test = pd.read_parquet('../resources/processed/test.parquet')

condensed_keys = [
    "_BMI5CAT",
    "_RFBMI5",
    "_CHLDCNT",
    "_EDUCAG",
    "_INCOMG1",
    "_RFMAM22",
    "_MAM5023",
    "_HADCOLN",
    "_CLNSCP1",
    "_HADSIGM",
    "_SGMSCP1",
    "_SGMS101",
    "_RFBLDS5",
    "_STOLDN1",
    "_VIRCOL1",
    "_SBONTI1",
    "_CRCREC2",
    "_SMOKER3",
    "_RFSMOK3",
    "_CURECI2",
    "_YRSSMOK",
    "_PACKYRS",
    "_YRSQUIT",
    "_SMOKGRP",
    "_LCSREC",
    "_RFBING6",
    "_RFDRHV8",
    "_FLSHOT7",
    "_PNEUMO3",
    "_AIDTST4"
]

looks_unimportant = [
    "FMONTH", "IDATE", "IMONTH",
      "IDAY", "IYEAR", "SEQNO", "_STATE",
        "DISPCODE", "_PSU"
]

_90_percent_nan = [
  'CHKHEMO3', 
  'HPVADSHT', 'COPDCOGH', 'COPDFLEM', 'COPDBRTH', 'COPDBTST', 'COPDSMOK', 'CNCRDIFF', 'CNCRAGE', 'CNCRTYP2', 'CSRVTRT3', 'CSRVDOC1', 'CSRVSUM', 'CSRVRTRN', 'CSRVINST', 'CSRVINSR', 'CSRVDEIN', 'CSRVCLIN', 'CSRVPAIN', 'CSRVCTL2', 'PSATEST1', 'PSATIME1', 'PCPSARS2', 'PSASUGST', 'PCSTALK1', 'CDHOUSE', 'MARIJAN1'
]

_50_percent_nan = [
    'NUMADULT', 'NUMMEN', 'NUMWOMEN', 
    # 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 
    'CHILDREN', 'SMOKDAY2', 'LCSFIRST', 
    'LCSLAST', 'LCSNUMCG', 'ALCDAY4', 
    'AVEDRNK3', 'DRNK3GE5', 'MAXDRNKS', 
    'FLSHTMY3', 'HIVTSTD3', 'COVIDFS1',
      'COVIDSE1', 'CIMEMLOS', '_PACKDAY'
      ]

to_drop = ['CVDINFR4', 'target'] + condensed_keys + _90_percent_nan + _50_percent_nan + looks_unimportant

important_columns = [
    "SEXVAR",
    "INCOME3",
    "RENTHOM1",
    "_STATE",
    "SMOKE100",
    "GENHLTH",  # General Health
    "PHYSHLTH",  # Number of Days Physical Health Not Good
    "MENTHLTH",  # Number of Days Mental Health Not Good
    "POORHLTH",  # Poor Physical or Mental Health
    "PRIMINSR",  # Primary Source of Health Insurance
    "PERSDOC3",  # Have Personal Health Care Provider
    "MEDCOST1",  # Could Not Afford To See Doctor
    "CHECKUP1",  # Length of Time Since Last Routine Checkup
    "EXERANY2",  # Exercise in Past 30 Days
    "SLEPTIM1",  # How Much Time Do You Sleep
    "LASTDEN4",  # Last Visited Dentist or Dental Clinic
    "RMVTETH4",  # Number of Permanent Teeth Removed
    "CVDCRHD4",  # Ever Diagnosed with Angina or Coronary Heart Disease
    "CVDSTRK3",  # Ever Diagnosed with a Stroke
    "ASTHMA3",  # Ever Told Had Asthma
    "CHCCOPD3",  # Ever Told You Had COPD, Emphysema, or Chronic Bronchitis
    "ADDEPEV3",  # Ever Told You Had a Depressive Disorder
    "CHCKDNY2",  # Ever Told You Have Kidney Disease
    "DIABETE4",  # Ever Told You Had Diabetes
    "_AGE80",  # Age of the respondent
    "_BMI5",  # Body Mass Index,
    "HAVARTH4",
    "CHCOCNC1",
    "CHCSCNC1"
]

y_train = df['target']
# X_train = df.drop(to_drop, axis=1)
X_train = df
X_train = X_train[important_columns]

y_test = df_test['target']
# X_test = df_test.drop(to_drop, axis=1)
X_test = df_test
X_test = X_test[important_columns]
