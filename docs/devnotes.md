# Developer Notes

## Technical Challenges

### Failing samples during autopipeline run
[2026-04-07] After reviewing the outputs from med-imagetools autopipeline run, eight samples were identified to have failed processing for legitimate issues with the CT or RTSTRUCT DICOM files. Currently procesing these manually by loading them into Slicer, as it seems to handle whatever these issues are, then saving them out as niftis. Will scrape the DICOM tags for the med-imagetools index next.

[2026-04-08]  
001 - CT_Ax_000
032 - CT_AXIAL.5.X.2.5_000
039 - CT_Chest.Abdo.Pel.C+..5.0..B31f_000
043 - CT_CHEST.ABD.PELVIS_000 - acquisitionNumber 2
059 - CT_Chest.Abdo.Pel.C+..5.0..B31f_000
064 - CT_ABDOMEN.AXIAL_000
072 - CT_5MM.AXIAL_000

082 - CT_A..5.X.2.5.-.v2_000 --> Brandon from QIPCM updated the CT to remove the spacing tag so it can be loaded, SeriesID is 2.16.840.1.114362.1.12400769.27653814164.733447348.858.2

Manually made the slicer index-simple

## Metadata Preprocessing

#### Cleaning of clinical data  
[2026-05-14] The clinical metdata file `Extracted Data - Disappearing Nodes Filtered.xlsx` shared by Justin has some calculations and the data definitions in the same sheet as the actual data. In order to process this, I made a copy of the file in `procdata` and removed the extra info. When loading the spreadsheet, I also found a hidden ID column A. This was all NaN, so I dropped it. 

I am saving out the fully preprocessed data as a `.csv` for easier downstream usage.

#### Stratification for model development
[2026-05-14] I think I'm going to handle the multiple lymph nodes by summing their volume and then including the number of lymph nodes as well, so will end up stratifying by relapse status, total lymph node volume, and number of lymph nodes.  

- The train_test_split function doesn't like the continuous MeshVolume value, so going to do above and below median

- Decided on a train-validation split of 80-20

[2026-07-13] When reprocessing the data for negative control analysis, discovered that AutoWATChmAN-082 was missed in original modelling. Have added it to the train/test split file manually in the validation cohort. 

Justin also pointed out that I should stratify by patient even for the radiomic models, so adding that in. 

## Analysis Notes
[2026-06-03] Jarvais trainer is working with a test_set = 0 (thanks Josh), so I'm getting results now. The models Jarvais uses are more complex, so for stuff like the vol_count and volume, should look at some linear/log regression classifiers

## Dependencies and Environment
[2026-07-13] Jarvais manual updates  
In autogluon_trainer.py I changed the following:  
```
from sklearn.model_selection import KFold, GroupKFold
...
def _train_autogluon_with_cv(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> tuple[TabularPredictor, pd.DataFrame, pd.Series]:

    groups = X_train['PatientID'] if 'PatientID' in X_train.columns else None
    X_train = X_train.drop(columns=['PatientID'], errors='ignore')
    kf = GroupKFold(n_splits=self.k_folds) #, shuffle=True, random_state=42)

    val_indices = []    

    for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train, groups)):
        X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]
```
I added the GroupKFold call so that I could group my patients together during k-fold cross validation and model training. I also drop the PatientID column so it isn't used as a variable for classification.


