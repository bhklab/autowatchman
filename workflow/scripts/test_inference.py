from damply import dirs
import pandas as pd
import numpy as np

from autogluon.tabular import TabularPredictor

PROJECT_ID = "PMCC_AutoWATChmAN"
TARGET = 'RELAPSE_STATUS'
MODEL_TYPE = 'clinical'
FEAT_NUM = 0
IMAGE_TYPE = 'original_full'

DATA_DIR = dirs.PROCDATA / PROJECT_ID
CLINICAL_PATH = DATA_DIR / 'metadata' / 'labelled_clinical_metadata_valid_0.2/csv'
RADIOMIC_PATH = DATA_DIR / 'features' / 'pyradiomics' / f'labelled_{IMAGE_TYPE}_linear_all_images_features_valid_0.2.csv'
FOUNDATION_PATH = DATA_DIR / 'features' / 'fmcib' / 'centroid_50_50_50' / f'labelled_{IMAGE_TYPE}_features_valid_0.2.csv'

feat_num = 'all' if FEAT_NUM == 0 else FEAT_NUM


MODEL_PATH = dirs.RESULTS / PROJECT_ID / "jarvais" / TARGET / f"{MODEL_TYPE}_{feat_num}" / f"trainer_{TARGET}" 
WEIGHTS_PATH = MODEL_PATH / "autogluon_models" / "autogluon_models_best_fold"      # your saved model
OUTPUT_PATH = MODEL_PATH / "test_inference" / IMAGE_TYPE
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --- Load model ---
predictor = TabularPredictor.load(WEIGHTS_PATH)

pred_feat_names = predictor.features()

for type_feature_set_path in [CLINICAL_PATH, RADIOMIC_PATH, FOUNDATION_PATH]:
    type_feature_set = pd.read_csv(type_feature_set_path, index_col = 0)

    if np.any([f in type_feature_set.columns for f in pred_feat_names]):
        pred_type_feature_set = type_feature_set[pred_feat_names]


