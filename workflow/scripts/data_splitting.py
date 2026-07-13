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

def split_sampleid_col(data_df: pd.DataFrame, sample_id_col: str = 'SampleID', sample_id_map: dict = {0:'SampleNumber', 1:'PatientID'}) -> pd.DataFrame:
    """Split the SampleID column into separate columns for PatientID and SampleNumber.

    Args:
        data_df (pd.DataFrame): The dataframe containing the SampleID column.
        sample_id_col (str, optional): The name of the SampleID column. Defaults to 'SampleID'.
        sample_id_map (dict, optional): A dictionary mapping the split ID columns to the desired column names. Defaults to {0:'SampleNumber', 1:'PatientID'}.
    
    Returns:
        pd.DataFrame: The dataframe with the SampleID column split into separate columns for PatientID and SampleNumber.
    """
    # Split the IDs into PatientID and SampleNumber columns, and rename the columns to match the clinical metadata
    split_ids = data_df[sample_id_col].str.split("__", expand=True)
    
    # sample_id_map should match the order used in med-imagetools autopipeline processing
    split_ids = split_ids.rename(columns=sample_id_map)
    
    # Add the split IDs to the dataframe
    labelled_data_df = pd.concat([split_ids, data_df], axis=1)

    return labelled_data_df


def label_radiomics(radiomic_feature_df: pd.DataFrame, 
                    split_labels: pd.Series,
                    sample_id_map: dict = {0:'SampleNumber', 1:'PatientID'}
                    ) -> pd.DataFrame:
    """Label the radiomic feature dataframe with the split labels and the PatientID and SampleNumber columns.

    Args:
        radiomic_feature_df (pd.DataFrame): The radiomic feature dataframe.
        split_labels (pd.Series): The split labels for each sample.
        sample_id_map (dict, optional): A dictionary mapping the split ID columns to the desired column names. Defaults to {0:'SampleNumber', 1:'PatientID'}.
    
    Returns:
        pd.DataFrame: The labelled radiomic feature dataframe with the split labels and separated PatientID and SampleNumber columns.
    """
    if 'PatientID' in radiomic_feature_df.columns or 'SampleNumber' in radiomic_feature_df.columns:
        logger.warning('SampleID column already split into PatientID and SampleNumber columns. Skipping split.')
        labelled_radiomic_feature_df = radiomic_feature_df
    else:
        if 'SampleID' in radiomic_feature_df.columns:
            logger.info('Splitting SampleID column into PatientID and SampleNumber columns.')
            labelled_radiomic_feature_df = split_sampleid_col(radiomic_feature_df, sample_id_col='SampleID', sample_id_map=sample_id_map)
        else:
            message = 'SampleID column not found in radiomic feature dataframe. Cannot split into PatientID and SampleNumber columns.'
            logger.exception(message)
            raise ValueError(message)


    # Create a unique LymphID for each lymph node by combining the PatientID and MaskID columns
    labelled_radiomic_feature_df['LymphID'] = labelled_radiomic_feature_df['PatientID'] + labelled_radiomic_feature_df['MaskID']
    
    # Index the dataframe by PatientID, SampleNumber, and MaskID to match the split metadata
    labelled_radiomic_feature_df = labelled_radiomic_feature_df.set_index(['PatientID', 'SampleNumber', 'MaskID'])
    
    # map the split column onto the radiomic stratification dataframe
    labelled_radiomic_feature_df = pd.merge(labelled_radiomic_feature_df, split_labels, left_index=True, right_index=True, how='inner')

    return labelled_radiomic_feature_df


def label_clinical(clinical_metadata_df: pd.DataFrame, 
                   split_labels: pd.Series
                   ) -> pd.DataFrame:
    """Label the clinical metadata dataframe with the split labels. 
    """
    if clinical_metadata_df.index.name != 'PatientID':
        message = 'Clinical metadata dataframe index is not named PatientID. Cannot merge with split labels.'
        logger.exception(message)
        raise ValueError(message)
    if split_labels.index.name != 'PatientID':
        message = 'Split labels index is not named PatientID. Cannot merge with clinical metadata dataframe.'
        logger.exception(message)
        raise ValueError(message)
    
    labelled_clinical_metadata_df = clinical_metadata_df.copy()
    labelled_clinical_metadata_df = pd.merge(labelled_clinical_metadata_df, split_labels, left_index=True, right_index=True, how='inner')

    return labelled_clinical_metadata_df


def main(
        dataset_name: str,
        clinical_metadata_path: Path | str,
        radiomic_feature_path: Path | str,
        clinical_stratification: list[str],
        radiomic_stratification: list[str] = None,
        sample_id_map: dict = {0:'SampleNumber', 1:'PatientID'},
        valid_size: float = 0.2,
        random_seed=10,
        ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    # load metadata file
    clinical_metadata_df = pd.read_csv(clinical_metadata_path, index_col=0)
    logger.info(f'Loaded clinical metadata from {clinical_metadata_path} with shape {clinical_metadata_df.shape}')
    radiomic_feature_df = pd.read_csv(radiomic_feature_path)#, index_col=0)
    logger.info(f'Loaded radiomic feature data from {radiomic_feature_path} with shape {radiomic_feature_df.shape}')
    radiomic_feature_df = split_sampleid_col(radiomic_feature_df, sample_id_col='SampleID', sample_id_map=sample_id_map)

    # Set the name of the clinical index to eventually match with radiomic feature ID label
    clinical_metadata_df.index.name = 'PatientID'

    # select out the stratification columns
    clinical_stratification_df = clinical_metadata_df[clinical_stratification]
    logger.info(f'Selected clinical stratification columns: {clinical_stratification} with shape {clinical_stratification_df.shape}')
    if radiomic_stratification is not None:
        # Select out the radiomic stratifying columns and the SampleID for grouping
        radiomic_stratification_df = radiomic_feature_df.loc[:, ['PatientID', *radiomic_stratification]]
        logger.info(f'Selected radiomic stratification columns: {radiomic_stratification} with shape {radiomic_stratification_df.shape}')

        # group the radiomic data by sampleID, add up the stratification values for each sample and save the total and the number of rows in the group
        agg_radiomic_stratification_df = radiomic_stratification_df.groupby(['PatientID']).agg(['sum', 'count'])
        agg_radiomic_stratification_df.columns = ['_'.join(col) for col in agg_radiomic_stratification_df.columns]
        logger.info(f'Aggregated radiomic stratification data with shape {agg_radiomic_stratification_df.shape}')

        # Find the median MeshVolume value, and make a binary column for samples above and below this median
        median_MeshVolume = agg_radiomic_stratification_df['original_shape_MeshVolume_sum'].median()
        agg_radiomic_stratification_df['above_median_MeshVolume'] = agg_radiomic_stratification_df['original_shape_MeshVolume_sum'] > median_MeshVolume

        # Remove the summed MeshVolume column, will mess up the stratification
        agg_radiomic_stratification_df = agg_radiomic_stratification_df.drop(columns=['original_shape_MeshVolume_sum'])
        logger.info(f'Dropped summed MeshVolume column, shape is now {agg_radiomic_stratification_df.shape}')

        # Strip the last five characters from the SampleID index to match the clinical metadata (removing the lymph node identifier)
        # agg_radiomic_stratification_df.index = agg_radiomic_stratification_df.index.str[:-5]
        # agg_radiomic_stratification_df.index.name = 'PatientID'
        
        # combine the stratification columns into one dataframe
        combined_strat_df = clinical_stratification_df.join(agg_radiomic_stratification_df, how='inner')
    else:
        combined_strat_df = clinical_stratification_df.copy()
    
    print(combined_strat_df.columns)
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
    labelled_radiomic_feature_df = label_radiomics(radiomic_feature_df, split_labels)

    # map the split column onto the radiomic stratification dataframe
    # split_ids = radiomic_feature_df.SampleID.str.split("_", expand=True)
    # split_ids = split_ids.rename(columns={0:'PatientID', 1:'SampleNumber'})

    # labelled_radiomic_feature_df = pd.concat([split_ids, radiomic_feature_df], axis=1)
    # labelled_radiomic_feature_df['LymphID'] = labelled_radiomic_feature_df['PatientID'] + labelled_radiomic_feature_df['MaskID']
    # labelled_radiomic_feature_df = labelled_radiomic_feature_df.set_index(['PatientID', 'SampleNumber', 'MaskID'])
    # labelled_radiomic_feature_df['split'] = split_labels
    
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
    labelled_clinical_metadata_df = pd.merge(labelled_clinical_metadata_df, split_labels, left_index=True, right_index=True, how='inner')

    return split_labels, labelled_clinical_metadata_df, labelled_radiomic_feature_df

    


if __name__ == '__main__':
    logger.info(f'Starting data_splitting script...')

    dataset_name = 'PMCC_AutoWATChmAN'
    image_type = 'original_full'

    clinical_metadata_path = dirs.PROCDATA / dataset_name / 'metadata' / 'cleaned_extracted_data_disappearing_nodes_filtered.csv'
    radiomic_feature_path = dirs.PROCDATA / dataset_name / 'features' / 'pyradiomics' / 'compiled_linear_all_images_features' / f'{image_type}_features.csv'

    clinical_stratification = ['RELAPSE_STATUS']
    radiomic_stratification = ['original_shape_MeshVolume']

    valid_size = .2

    label_file_path = dirs.PROCDATA / dataset_name / 'metadata' / f'train_valid_{valid_size}_labels.csv'

    if label_file_path.exists():
        logger.info('Split labels already generated. Loading existing file.')
        labels = pd.read_csv(label_file_path, index_col=0)
        clinical_metadata_df = pd.read_csv(clinical_metadata_path, index_col=0)
        clinical_metadata_df.index.name = 'PatientID'
        radiomic_feature_df = pd.read_csv(radiomic_feature_path)

        lbl_clinical = label_clinical(clinical_metadata_df, labels)
        lbl_radiomic = label_radiomics(radiomic_feature_df, labels)
    
    else:
        logger.info(f'No split labels found at {label_file_path}. Running function to generate them.')
        labels, lbl_clinical, lbl_radiomic = main(dataset_name, clinical_metadata_path, radiomic_feature_path, clinical_stratification, radiomic_stratification, valid_size=valid_size, random_seed=42)
        labels.to_csv(label_file_path, na_rep='NA')
    

    lbl_clinical.to_csv(dirs.PROCDATA / dataset_name / 'metadata' / f'labelled_clinical_metadata_valid_{valid_size}.csv', na_rep='NA')
    lbl_radiomic.to_csv(dirs.PROCDATA / dataset_name / 'features' / 'pyradiomics' / f'labelled_{image_type}_linear_all_images_features_valid_{valid_size}.csv', na_rep='NA')
    