from damply import dirs
import pandas as pd
import matplotlib.pyplot as plt

from jarvais.analyzer import Analyzer
from jarvais.trainer import TrainerSupervised

from autogluon.tabular import TabularPredictor

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def swap_patient_to_lymph_idx(df, lymph_index):
    """
    Swap the index of the dataframe from PatientID to LymphID using the provided lymph_index series. Keep PatientID as a column in the dataframe.
    
    Parameters:
    df (pd.DataFrame): The dataframe with PatientID as index.
    lymph_index (pd.Series): A series mapping PatientID to LymphID.
    
    Returns:
    pd.DataFrame: The dataframe with LymphID as index.
    """
    dfc = df.copy()
    dfc = dfc.reset_index(drop=False)  # Keep PatientID as a column
    dfc.index = lymph_index
    return dfc


PROJECT_ID = "PMCC_AutoWATChmAN"
MODEL_TYPE = 'clinicoradiomic'
TARGET = 'RELAPSE_STATUS'
FEAT_NUM = 16
REDUCTION_METHOD = 'mrmr'
IMAGE_TYPE = 'shuffled_non_roi'
STRATIFY_ON = 'PatientID'

data_dir = dirs.PROCDATA / PROJECT_ID
output_dir = dirs.RESULTS / PROJECT_ID / "jarvais"

if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

full_clinical = pd.read_csv(data_dir / 'metadata' / 'labelled_clinical_metadata_valid_0.2.csv', index_col=0)
clinical = full_clinical[full_clinical['split']=='train']
clin_cat_cols = ['LVI', 'RETE_INV', 'LATERAL']
clin_cont_cols = ['AGE_DIAG', 'HIST_SEM', 'HIST_EC', 'HIST_CHOR', 'HIST_YOLK', 'HIST_TERA']
clin_pred_cols = clin_cat_cols + clin_cont_cols

full_features = pd.read_csv(data_dir / 'features' / 'pyradiomics' / f'labelled_{IMAGE_TYPE}_linear_all_images_features_valid_0.2.csv', index_col=0)
features = full_features[full_features['split']=='train']
features = features.drop('split', axis=1)
lymph_index = features['LymphID']
features = features.filter(regex=r"^original_*")

# Initilize both of these as empty, set in match case if used
cat_cols = [TARGET]
cont_cols = []
match MODEL_TYPE:
    case 'clinical':
        train_data = clinical.loc[:, [TARGET] + clin_pred_cols]
        cat_cols += clin_cat_cols
        cont_cols += clin_cont_cols
        STRATIFY_ON = None
    case 'vol_count':
        train_data = clinical.loc[:, [TARGET] + ['LN_NUM']]
        cont_cols += ['LN_NUM']
        STRATIFY_ON = None
    case 'volume':
        train_data = pd.merge(clinical.loc[:, [TARGET]], features.loc[:, ['original_shape_MeshVolume']], left_index=True, right_index=True)
        train_data = swap_patient_to_lymph_idx(train_data, lymph_index)
        cont_cols += ['original_shape_MeshVolume']
    case 'clinvol':
        train_data = pd.merge(clinical.loc[:, [TARGET] + clin_pred_cols], features.loc[:, ['original_shape_MeshVolume']], left_index=True, right_index=True)
        train_data = swap_patient_to_lymph_idx(train_data, lymph_index)
        cat_cols += clin_cat_cols
        cont_cols += clin_cont_cols + ['original_shape_MeshVolume']
    case 'radiomic':
        train_data = pd.merge(clinical.loc[:, [TARGET]], features, left_index=True, right_index=True)
        train_data = swap_patient_to_lymph_idx(train_data, lymph_index)
        cont_cols += features.columns.to_list()
    case 'radiomic_no_shape':
        features_no_shape = features.filter(regex=r"^(?!original_shape).*", axis=1)
        train_data = pd.merge(clinical.loc[:, [TARGET]], features_no_shape, left_index=True, right_index=True)
        train_data = swap_patient_to_lymph_idx(train_data, lymph_index)
        cont_cols += features_no_shape.columns.to_list()
    case 'clinicoradiomic':
        train_data = pd.merge(clinical.loc[:, [TARGET] + clin_pred_cols], features, left_index=True, right_index=True)
        train_data = swap_patient_to_lymph_idx(train_data, lymph_index)
        cat_cols += clin_cat_cols
        cont_cols += clin_cont_cols + features.columns.to_list()
    case 'clinicoradiomic_no_shape':
        features_no_shape = features.filter(regex=r"^(?!original_shape).*", axis=1)
        train_data = pd.merge(clinical.loc[:, [TARGET] + clin_pred_cols], features_no_shape, left_index=True, right_index=True)
        train_data = swap_patient_to_lymph_idx(train_data, lymph_index)
        cat_cols += clin_cat_cols
        cont_cols += clin_cont_cols + features_no_shape.columns.to_list()

for col in cont_cols:
    train_data[col] = train_data[col].fillna(0)

if FEAT_NUM > 0 and REDUCTION_METHOD is None:
    raise ValueError("If FEAT_NUM is greater than 0, REDUCTION_METHOD must be specified.")
elif FEAT_NUM == 0:
    REDUCTION_METHOD = None
    MODEL_TYPE = MODEL_TYPE + "_all"
    output_dir = output_dir / TARGET / MODEL_TYPE
else:
    MODEL_TYPE = f"{MODEL_TYPE}_{FEAT_NUM}"
    output_dir = output_dir / TARGET / MODEL_TYPE


if IMAGE_TYPE == 'original_full':
    # Run Analyzer
    print(f"Initializing Analyzer for {TARGET}...")
    analyzer = Analyzer(
        train_data,
        output_dir = output_dir / f"analyzer_{TARGET}",
        categorical_columns = cat_cols,
        continuous_columns = cont_cols,
        target_variable = TARGET,
        task="classification",
    )

    # Drop multiplotting, expensive operation
    analyzer.settings.visualization.plots.remove('multiplot')
    print(f"Running Analyzer for {MODEL_TYPE}_{FEAT_NUM} {TARGET}...")
    analyzer.run()


    print(f"Initializing Trainer for {TARGET}...")
    # Run Trainer
    trainer = TrainerSupervised(
        output_dir= output_dir / f"trainer_{TARGET}",
        target_variable = TARGET,
        task = 'binary',
        # stratify_on= STRATIFY_ON,
        test_size=0.0,
        k_folds=5,
        reduction_method=REDUCTION_METHOD,
        keep_k=FEAT_NUM,
        explain = False,
        random_state=42
    )

    # print(trainer)
    analyzer.data[TARGET] = analyzer.data[TARGET].astype(int)

    print(f"Running Trainer for {TARGET}...")
    trainer.run(analyzer.data)
else:
    print("Only training on original images, not negative controls.")
    

# Inference
MODEL_PATH = dirs.RESULTS / PROJECT_ID / "jarvais" / TARGET / MODEL_TYPE / f"trainer_{TARGET}" 
WEIGHTS_PATH = MODEL_PATH / "autogluon_models" / "autogluon_models_best_fold"      # your saved model
OUTPUT_PATH = MODEL_PATH / "inference" / IMAGE_TYPE
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --- Load model ---
predictor = TabularPredictor.load(WEIGHTS_PATH)

if IMAGE_TYPE == 'original_full':
    feature_importance = predictor.feature_importance(train_data)
    feature_importance.to_csv(MODEL_PATH / "feature_importance.csv", index=True)

inf_data = train_data.drop(TARGET, axis=1)

# -- Run inference ---
try:
    preds = predictor.predict(inf_data)
except Exception as e:
    print("Error during prediction:", e)
    raise

# --- Probabilities (classification only) ---
probs = None
try:
    probs = predictor.predict_proba(inf_data)
except Exception:
    pass

probs.to_csv(OUTPUT_PATH / "predicted_probabilities.csv", index=True)

conf_mtx = confusion_matrix(train_data[TARGET], preds)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mtx)
disp.plot()
plt.grid(visible=False)
plt.title(f"{MODEL_TYPE} {TARGET} {IMAGE_TYPE}")
plt.savefig(OUTPUT_PATH / "confusion_matrix.png")



