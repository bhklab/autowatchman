from damply import dirs
import pandas as pd

from jarvais.analyzer import Analyzer
from jarvais.trainer import TrainerSupervised

data_dir = dirs.PROCDATA / "PMCC_AutoWATChmAN"
output_dir = dirs.RESULTS / "PMCC_AutoWATChmAN" / "jarvais"

if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

full_clinical = data_dir / 'metadata' / 'labelled_clinical_metadata_valid_0.2.csv'
clinical = full_clinical[full_clinical['split']=='train']

full_features = data_dir / 'features' / 'pyradiomics' / 'labelled_linear_all_images_features_valid_0.2.csv'
features = full_features[full_features['split']=='train']

print(len(clinical))
print(len(features))