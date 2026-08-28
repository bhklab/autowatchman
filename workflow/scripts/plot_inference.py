from damply import dirs
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

PROJECT_ID = "PMCC_AutoWATChmAN"
MODEL_TYPE = 'set_clinicoradiomic'
TARGET = 'RELAPSE_STATUS'
FEAT_NUM = 'all'
REDUCTION_METHOD = 'mrmr'
IMAGE_TYPE = 'randomized_non_roi'

# --- Paths ---
MODEL_PATH = dirs.RESULTS / PROJECT_ID / "jarvais" / TARGET / f"{MODEL_TYPE}_{FEAT_NUM}" / f"trainer_{TARGET}"
OUTPUT_PATH = MODEL_PATH / "inference" / IMAGE_TYPE

input_data_path = MODEL_PATH / "data" / "input_data.csv"
pred_data_path = OUTPUT_PATH / "predictions.csv"

input_data = pd.read_csv(input_data_path)
pred_data = pd.read_csv(pred_data_path)

# pats = 

# conf_mtx = confusion_matrix(input_data.loc[:,TARGET], pred_data[TARGET])
disp = ConfusionMatrixDisplay.from_predictions(
    y_true = input_data.loc[:,TARGET],
    y_pred = pred_data[TARGET],
    )
disp.plot(im_kw={'vmin': 0, 'vmax': 55})
plt.grid(visible=False)
plt.title(f"{MODEL_TYPE}_{FEAT_NUM} {TARGET} {IMAGE_TYPE}")
plt.savefig(OUTPUT_PATH / "confusion_matrix.png")
