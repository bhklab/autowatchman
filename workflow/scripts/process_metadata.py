"""
This script processes the metadata for the PMCC AutoWATChmAN project. It checks for the existence of a cleaned CSV metadata file, and if it doesn't exist, it looks for an Excel file to convert. The Excel file is expected to have been manually edited to remove extra calculations and class labels around the data. The script then saves the cleaned metadata as a CSV file for further use in the project.
"""

import logging
import pandas as pd
from damply import dirs

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main() -> None:
    # load metadata file
    metadata_file_path = dirs.PROCDATA / 'PMCC_AutoWATChmAN' / 'metadata' / 'cleaned_extracted_data_disappearing_nodes_filtered.csv'

    if not metadata_file_path.exists():
         logger.info(f'Clean metadata file not found at {metadata_file_path}. Looking for Excel file to convert...')

         # This file has been manually edited to remove extra calculations and class labels around the data
         temp_metadata_file = dirs.PROCDATA / 'PMCC_AutoWATChmAN' / 'metadata' / 'cleaned_extracted_data_disappearing_nodes_filtered.xlsx'
         if not temp_metadata_file.exists():
             logger.error(f'Excel metadata file not found at {temp_metadata_file}. Please ensure the file exists and is in the correct location.')
             return
         else:
             logger.info(f'Excel metadata file found at {temp_metadata_file}. Converting to CSV...')
             metadata_df = pd.read_excel(temp_metadata_file) 
             metadata_df = metadata_df.drop(columns=['ID'])
             metadata_df.to_csv(dirs.PROCDATA / 'PMCC_AutoWATChmAN' / 'metadata' / 'cleaned_extracted_data_disappearing_nodes_filtered.csv', index=False)

    else:
        logger.info(f'Clean metadata file found at {metadata_file_path}. Loading CSV...')
        metadata_df = pd.read_csv(metadata_file_path)
    return


if __name__ == '__main__':
    logger.info(f'Starting process_metadata script...')
    main()