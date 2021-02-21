import hydra
import logging
import sys
import os
import os.path
import zipfile

from torch import hub
from asteroid.data import TimitDataset
from pathlib import PurePath

logger = logging.getLogger(__name__)

NOISES_URL = 'https://zenodo.org/record/4553667/files/all_drone_noises.zip?download=1'

def download_and_unpack_noises(directory):
    directory = PurePath(directory)
    os.makedirs(directory, exist_ok=True)
    if os.path.isdir(directory / 'noises-train-drones') and os.path.isdir(directory / 'noises-test-drones'):
        logger.info('Noises data seems to be already loaded')
        return
    
    logger.info('Downloading and extracting noises...')
    zip_path = directory / 'noises.zip'
    hub.download_url_to_file(NOISES_URL, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(directory)
        
    os.remove(zip_path)
    logger.info('Done')

    
def download_datasets(args):
    logger.info('Loading datasets')
    TimitDataset.download(args.dset.timit, sample_rate=args.sample_rate)
    download_and_unpack_noises(args.dset.noises)
    

@hydra.main(config_path='conf', config_name='config')
def main(args):
    download_datasets(args)
    
if __name__ == "__main__":
    main()
