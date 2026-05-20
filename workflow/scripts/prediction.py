from damply import dirs
import pandas as pd

# from jarvais.analyzer import Analyzer
# from jarvais.trainer import TrainerSupervised

data_dir = dirs.PROCDATA / "PMCC_AutoWATChmAN"
output_dir = dirs.RESULTS / "PMCC_AutoWATChmAN" / "jarvais"

if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

full_clinical = pd.read_csv(data_dir / 'metadata' / 'labelled_clinical_metadata_valid_0.2.csv', index_col=0)
clinical = full_clinical[full_clinical['split']=='train']
clin_pred_cols = ['AGE_DIAG', 'LVI', 'RETE_INV', 'TMRSZE', 'HIST_SEM', 'HIST_EC', 'HIST_CHOR', 'HIST_YOLK', 'HIST_TERA']

full_features = pd.read_csv(data_dir / 'features' / 'pyradiomics' / 'labelled_linear_all_images_features_valid_0.2.csv', index_col=0)
features = full_features[full_features['split']=='train']
features = features.drop('split', axis=1)
lymph_index = features['LymphID']
features = features.filter(regex=r"^original_*")

MODEL_TYPE = 'clinvol'
TARGET = 'RELAPSE_STATUS'
FEAT_NUM = 15
REDUCTION_METHOD = 'mrmr'
output_dir = output_dir / f"{MODEL_TYPE}_{FEAT_NUM}" / TARGET

match MODEL_TYPE:
    case 'clinical':
        train_data = clinical.loc[:, [TARGET] + clin_pred_cols]
    case 'vol_count':
        train_data = clinical.loc[:, [TARGET, 'LN_NUM']]
    case 'volume':
        train_data = pd.merge(clinical.loc[:, [TARGET]], features.loc[:, 'original_shape_MeshVolume'], left_index=True, right_index=True)
        train_data.index = lymph_index
    case 'clinvol':
        train_data = pd.merge(clinical.loc[:, [TARGET] + clin_pred_cols], features.loc[:, 'original_shape_MeshVolume'], left_index=True, right_index=True)
        train_data.index = lymph_index
    case 'radiomic':
        train_data = pd.merge(clinical.loc[:, [TARGET]], features, left_index=True, right_index=True)
        train_data.index = lymph_index
    case 'clinicoradiomic':
        train_data = pd.merge(clinical.loc[:, [TARGET] + clin_pred_cols], features, left_index=True, right_index=True)
        train_data.index = lymph_index

print(train_data.head())



# print(f"Initializing Analyzer for {TARGET}...")
# # Run Analyzer
# analyzer = Analyzer(
#     train_data,
#     output_dir = output_dir / f"analyzer_{TARGET}",
#     categorical_columns = [TARGET],
#     target_variable = TARGET,
#     task="classification"
# )

# # Drop multiplotting, expensive operation
# analyzer.settings.visualization.plots.remove('multiplot')

# # # print(f"Running Analyzer for {TARGET}...")
# # # analyzer.run()


# print(f"Initializing Trainer for {TARGET}...")
# # Run Trainer
# trainer = TrainerSupervised(
#     output_dir= output_dir / f"trainer_{TARGET}",
#     target_variable = TARGET,
#     task = 'binary',
#     stratify_on= TARGET,
#     test_size=0.2,
#     k_folds=5,
#     reduction_method=reduction_method,
#     keep_k=feature_num,
#     explain = True
# )

# print(trainer)
# analyzer.data[TARGET] = analyzer.data[TARGET].astype(int)

# print(f"Running Trainer for {TARGET}...")
# trainer.run(analyzer.data)