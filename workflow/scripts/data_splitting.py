import logging
import pandas as pd
from damply import dirs
from pathlib import Path
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    filename=dirs.LOGS / "data_splitting.log"
)

logger = logging.getLogger(__name__)


def main(
        dataset_name: str,
        clinical_metadata_path: Path | str,
        radiomic_feature_path: Path | str,
        clinical_stratification: list[str],
        radiomic_stratification: list[str] = None,
        valid_size: float = 0.2,
        random_seed=10,
        ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    # load metadata file
    clinical_metadata_df = pd.read_csv(clinical_metadata_path, index_col=0)
    radiomic_feature_df = pd.read_csv(radiomic_feature_path)#, index_col=0)

    # Set the name of the clinical index to eventually match with radiomic feature ID label
    clinical_metadata_df.index.name = 'PatientID'

    # select out the stratification columns
    clinical_stratification_df = clinical_metadata_df[clinical_stratification]
    if radiomic_stratification is not None:
        # Select out the radiomic stratifying columns and the SampleID for grouping
        radiomic_stratification_df = radiomic_feature_df.loc[:, ['SampleID', *radiomic_stratification]]

        # group the radiomic data by sampleID, add up the stratification values for each sample and save the total and the number of rows in the group
        agg_radiomic_stratification_df = radiomic_stratification_df.groupby(['SampleID']).agg(['sum', 'count'])
        agg_radiomic_stratification_df.columns = ['_'.join(col) for col in agg_radiomic_stratification_df.columns]

        # Find the median MeshVolume value, and make a binary column for samples above and below this median
        median_MeshVolume = agg_radiomic_stratification_df['original_shape_MeshVolume_sum'].median()
        agg_radiomic_stratification_df['above_median_MeshVolume'] = agg_radiomic_stratification_df['original_shape_MeshVolume_sum'] > median_MeshVolume

        # Remove the summed MeshVolume column, will mess up the stratification
        agg_radiomic_stratification_df = agg_radiomic_stratification_df.drop(columns=['original_shape_MeshVolume_sum'])

        # Strip the last five characters from the SampleID index to match the clinical metadata (removing the lymph node identifier)
        agg_radiomic_stratification_df.index = agg_radiomic_stratification_df.index.str[:-5]
        agg_radiomic_stratification_df.index.name = 'PatientID'
        
    # combine the stratification columns into one dataframe
    combined_strat_df = clinical_stratification_df.join(agg_radiomic_stratification_df, how='inner')
    
    # Perform train valid splitting
    tr_samples, valid_samples = train_test_split(
        combined_strat_df,
        test_size=valid_size,
        random_state = random_seed,
        shuffle = True,
        stratify = combined_strat_df
    )

    # add split labels to each subset and recombine
    tr_samples['split'] = 'train'
    valid_samples['split'] = 'valid'

    all_samples = pd.concat([tr_samples,valid_samples], axis=0)
    split_labels = all_samples['split'].sort_index()

    # confirmation of distribution split plots
    # Set up directory to save figures to
    fig_dir = dirs.PROCDATA / dataset_name / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # histogram plots for each category
    dist_fig, dist_axes = plt.subplots(nrows=1, ncols=len(all_samples.columns), figsize=(15,7))

    for col_idx, col in enumerate(all_samples.columns):
        sns.histplot(
            all_samples, 
            x = col,
            stat = 'count',
            ax=dist_axes[col_idx],
            multiple='dodge',
            shrink=.8,
            hue='split',
            discrete=True,
            hue_order = ['train', 'valid']
        )

    dist_fig.savefig(fig_dir / f'split_dist_fig_valid_{valid_size}.png', bbox_inches='tight')

    # scatter plot for volume of lymph nodes
    # map the split column onto the radiomic stratification dataframe
    split_ids = radiomic_feature_df.SampleID.str.split("_", expand=True)
    split_ids = split_ids.rename(columns={0:'PatientID', 1:'SampleNumber'})

    labelled_radiomic_feature_df = pd.concat([split_ids, radiomic_feature_df], axis=1)
    labelled_radiomic_feature_df['LymphID'] = labelled_radiomic_feature_df['PatientID'] + labelled_radiomic_feature_df['MaskID']
    labelled_radiomic_feature_df = labelled_radiomic_feature_df.set_index(['PatientID', 'SampleNumber', 'MaskID'])
    labelled_radiomic_feature_df['split'] = split_labels
    
    scatter_fig = plt.figure(figsize=(8,5))
    scatter_fig.add_axes(
        sns.scatterplot(
            labelled_radiomic_feature_df.sort_values('original_shape_MeshVolume'),
            x = 'LymphID',
            y = 'original_shape_MeshVolume',
            hue = 'split',
            hue_order = ['train', 'valid']
        )
    )
    scatter_fig.axes[0].tick_params(axis='x', labelbottom=False)

    scatter_fig.savefig(fig_dir / f'split_scatter_fig_valid_{valid_size}.png', bbox_inches='tight')

    vol_dist_fig = plt.figure(figsize=(8,7))
    vol_dist_fig.add_axes(
        sns.histplot(
            labelled_radiomic_feature_df,
            x = 'original_shape_MeshVolume',
            stat = 'count',
            multiple = 'layer',
            hue = 'split',
            hue_order = ['train', 'valid']
        )
    )

    vol_dist_fig.savefig(fig_dir / f'vol_hist_fig_valid_{valid_size}.png', bbox_inches='tight')

    labelled_clinical_metadata_df = clinical_metadata_df.copy()
    labelled_clinical_metadata_df['split'] = split_labels

    return split_labels, labelled_clinical_metadata_df, labelled_radiomic_feature_df

    


if __name__ == '__main__':
    logger.info(f'Starting data_splitting script...')

    dataset_name = 'PMCC_AutoWATChmAN'

    clinical_metadata_path = dirs.PROCDATA / dataset_name / 'metadata' / 'cleaned_extracted_data_disappearing_nodes_filtered.csv'
    radiomic_feature_path = dirs.PROCDATA / dataset_name / 'features' / 'pyradiomics' / 'original_512_512_n' / 'original_full_features.csv'

    clinical_stratification = ['RELAPSE_STATUS']
    radiomic_stratification = ['original_shape_MeshVolume']

    valid_size = .2

    labels, lbl_clinical, lbl_radiomic = main(dataset_name, clinical_metadata_path, radiomic_feature_path, clinical_stratification, radiomic_stratification, valid_size=valid_size, random_seed=42)

    labels.to_csv(dirs.PROCDATA / dataset_name / 'metadata' / f'train_valid_{valid_size}_labels.csv')
    lbl_clinical.to_csv(dirs.PROCDATA / dataset_name / 'metadata' / f'labelled_clinical_metadata_valid_{valid_size}.csv')
    lbl_radiomic.to_csv(dirs.PROCDATA / dataset_name / 'features' / 'pyradiomics' / f'labelled_linear_all_images_features_valid_{valid_size}.csv')