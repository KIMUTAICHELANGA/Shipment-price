import os
import urllib.request as request
import zipfile
from src.MLProject import logger
from src.MLproject.utils.common import get_size
from pathlib import
from src.MLProject.entity.config_entity import ConfigEntity

class DataIngestion:
    def __init__(self, config: ConfigEntity):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.raw_data_dir):
            os.makedirs(self.config.raw_data_dir, exist_ok=True)
            logger.info(f"Creating directory; {self.config.raw_data_dir}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.raw_data_dir, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
