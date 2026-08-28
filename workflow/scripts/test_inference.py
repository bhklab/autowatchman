from damply import dirs
import pandas as pd
from functools import reduce
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor

PROJECT_ID = "PMCC_AutoWATChmAN"
TARGET = 'RELAPSE_STATUS'
MODEL_TYPE = 'clinicoradiomic'
FEAT_NUM = 16
IMAGE_TYPE = 'original_full'
SPLIT = 'valid'

DATA_DIR = dirs.PROCDATA / PROJECT_ID
CLINICAL_PATH = DATA_DIR / 'metadata' / 'labelled_clinical_metadata_valid_0.2.csv'
RADIOMIC_PATH = DATA_DIR / 'features' / 'pyradiomics' / f'labelled_{IMAGE_TYPE}_linear_all_images_features_valid_0.2.csv'
FOUNDATION_PATH = DATA_DIR / 'features' / 'fmcib' / 'centroid_50_50_50' / f'labelled_{IMAGE_TYPE}_features_valid_0.2.csv'

feat_num = 'all' if FEAT_NUM == 0 else FEAT_NUM


MODEL_PATH = dirs.RESULTS / PROJECT_ID / "archive_jarvais" / TARGET / f"{MODEL_TYPE}_{feat_num}" / f"trainer_{TARGET}" 
WEIGHTS_PATH = MODEL_PATH / "autogluon_models" / "autogluon_models_best_fold"      # your saved model
OUTPUT_PATH = MODEL_PATH / "test_inference" / IMAGE_TYPE
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --- Load model ---
predictor = TabularPredictor.load(WEIGHTS_PATH)

# Get list of features used in the model
pred_feat_names = [TARGET] + predictor.features()
inf_type_feat_sets = []

# Look for the prediction features in the clinical, radiomic, and foundation feature sets
for type_feature_set_path in [CLINICAL_PATH, RADIOMIC_PATH, FOUNDATION_PATH]:
    type_feature_set = pd.read_csv(type_feature_set_path, index_col = 0)
    # Get just the rows corresponding to the specified split (train or validation)
    type_feature_set = type_feature_set.loc[type_feature_set['split'] == SPLIT]
    
    # Get the intersection of the pred_feat_names and the columns of the type_feature_set
    int_feat_names = type_feature_set.columns.intersection(pred_feat_names).to_list()
    if len(int_feat_names) > 0:
        print(f"Found prediction features in the {type_feature_set_path.name} feature set.")
        inf_type_feature_set = type_feature_set[int_feat_names]
        # Add this set of features to the list of prediction feature sets
        inf_type_feat_sets.append(inf_type_feature_set)

    # Remove the found features from the list of prediction features
    pred_feat_names = [feat for feat in pred_feat_names if feat not in int_feat_names]
    if len(pred_feat_names) == 0:
        print("All prediction features have been found. Skipping remaining feature sets.")
        break

# Merge all of the feature sets together into a single dataframe, using the index (patientID) as the key
inf_feats = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), inf_type_feat_sets)
# Get the ground truth target prediction values and remove the target column from the inference features
target_gt = inf_feats.pop(TARGET)
target_gt.name = f"gt_{TARGET}"

# -- Run inference ---
try:
    preds = predictor.predict(inf_feats)
    labelled_preds = pd.concat([preds, target_gt], axis=1)# columns=[f"pred_{TARGET}", f"gt_{TARGET}"])
    labelled_preds.rename(columns={TARGET: f"pred_{TARGET}"}, inplace=True)
except Exception as e:
    print("Error during prediction:", e)
    raise

# --- Probabilities (classification only) ---
probs = None
try:
    probs = predictor.predict_proba(inf_feats)
except Exception:
    pass


# auc, f1_score, auprc, 
metrics = predictor.evaluate_predictions(y_true=target_gt, y_pred=probs, auxiliary_metrics=True, display=True)
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=[f"{MODEL_TYPE}_{feat_num}"])
metrics_df.to_csv(OUTPUT_PATH / "metrics.csv", index=True)


labelled_preds.to_csv(OUTPUT_PATH / "predictions.csv", index=True)
if probs is not None:
    probs[f"gt_{TARGET}"] = target_gt
    probs.to_csv(OUTPUT_PATH / "predicted_probabilities.csv", index=True)


# Plot and save confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(
    y_true = labelled_preds[f"gt_{TARGET}"],
    y_pred = labelled_preds[f"pred_{TARGET}"],
    )
disp.plot(im_kw={'vmin': 0, 'vmax': 15})
plt.grid(visible=False)
plt.title(f"{MODEL_TYPE}_{FEAT_NUM} {TARGET} {IMAGE_TYPE}")
plt.savefig(OUTPUT_PATH / "confusion_matrix.png")