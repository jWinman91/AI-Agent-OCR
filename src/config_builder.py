import yaml
import os
from string import Template
from typing import Dict

from src.utils import sqlite_db_handler
from src.utils.data_models import ConfigModelDB, ConfigModelDBReduced


class ConfigBuilder:
    SYSTEM_PROMPT = "system_prompt"
    USER_PROMPT = "system_prompt_user"
    PROMPT_TEMPLATE = "system_prompt_template"
    CONFIG_DB_PATH = "config.db"

    def __init__(self, config_file_path: str, table_name: str = "configs"):
        self._sqlite_handler = sqlite_db_handler.SqliteDBHandler(table_name)

        if not os.path.exists(self.CONFIG_DB_PATH):
            config_template = self.load_yaml(config_file_path)
            for config_name, config in config_template.items():
                self.write_config_to_db(config_name, config)

    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, ConfigModelDB]:
        """
        Loads in a system prompt (and additional parameters) for the LLM and returns them as a tuple.
        :param config_path: path to yaml file
        :return: Tuple containing the system prompt and additional parameters
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def load_config_from_db(self, config_name: str) -> ConfigModelDB:
        config = self._sqlite_handler.get_config(config_name)
        return ConfigModelDB(**config)

    def load_all_configs_from_db(self) -> Dict[str, ConfigModelDB]:
        config_names = self._sqlite_handler.get_all_config_names()

        all_configs = {}
        for config_name in config_names:
            all_configs[config_name] = self.load_config_from_db(config_name)

        return all_configs

    def write_config_to_db(self, config_name: str, config: ConfigModelDB) -> bool | None:
        return self._sqlite_handler.add_config(config.model_dump(), config_name)

    def update_config_in_db(self, config_name: str, config: ConfigModelDBReduced) -> bool | None:
        return self._sqlite_handler.update_config(config.model_dump(), config_name)

    def delete_config_from_db(self, config_name: str) -> bool | None:
        return self._sqlite_handler.delete_config(config_name)

    def build_config(self) -> dict:
        configs_db = self.load_all_configs_from_db()
        configs = {}
        for config_name, config_db in configs_db.items():
            config = {}
            prompt_template = Template(config_db.model_dump()[self.PROMPT_TEMPLATE])
            user_part = config_db.model_dump()[self.USER_PROMPT]
            config[self.SYSTEM_PROMPT] = prompt_template.substitute(user_part=user_part)
            configs[config_name] = config
        return configs