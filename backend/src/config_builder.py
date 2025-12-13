import os
from string import Template
from typing import Any, Dict

import yaml
from src.utils import sqlite_db_handler
from src.utils.data_models import ConfigModelBuild, ConfigModelDB, ConfigModelUser


class ConfigBuilder:
    SYSTEM_PROMPT = "system_prompt"
    USER_PROMPT = "system_prompt_user"
    PROMPT_TEMPLATE = "system_prompt_template"
    CONFIG_DB_PATH = "config.db"

    def __init__(
        self,
        config_file_path: str,
        table_name: str = "configs",
    ) -> None:
        """
        Initializes the ConfigBuilder class.

        :param config_file_path: Path to the YAML configuration file.
        :param table_name: Name of the database table to store configurations.
        """
        self.SQLITE_HANDLER = sqlite_db_handler.SqliteDBHandler(table_name)
        self.CONFIG_TEMPLATE = self._load_yaml(config_file_path)

        if not os.path.exists(self.CONFIG_DB_PATH):
            for config_name, config in self.CONFIG_TEMPLATE.items():
                self.write_config_to_db(config_name, ConfigModelDB(**config))

    @staticmethod
    def _load_yaml(config_path: str) -> Dict[str, Any]:
        """
        Loads in a system prompt (and additional parameters) for the LLM and returns
        them as a tuple.

        :param config_path: path to yaml file
        :return: Tuple containing the system prompt and additional parameters
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        if config is None:
            return {}
        return dict(config)

    def load_config_from_db(self, config_name: str) -> ConfigModelDB:
        """
        Loads a configuration from the database by its name.

        :param config_name: Name of the configuration to load.
        :return: ConfigModelDB instance containing the configuration data.
        """
        config = self.SQLITE_HANDLER.get_config(config_name)
        return ConfigModelDB(**config)

    def load_all_configs_from_db(self) -> Dict[str, ConfigModelDB]:
        """
        Loads all configurations from the database.

        :return: Dictionary mapping configuration names to ConfigModelDB instances.
        """
        config_names = self.SQLITE_HANDLER.get_all_config_names()

        all_configs = {}
        for config_name in config_names:
            all_configs[config_name] = self.load_config_from_db(config_name)

        return all_configs

    def write_config_to_db(
        self, config_name: str, config: ConfigModelDB
    ) -> bool | None:
        """
        Writes a configuration to the database.

        :param config_name: Name of the configuration to write.
        :param config: ConfigModelDB instance containing the configuration data.
        :return: True if the operation was successful, False otherwise.
        """
        return self.SQLITE_HANDLER.add_config(config.model_dump(), config_name)

    def update_config_in_db(
        self, config_name: str, config: ConfigModelUser
    ) -> bool | None:
        """
        Updates a configuration in the database.

        :param config_name: Name of the configuration to update.
        :param config: ConfigModelDBReduced instance containing the configuration data.
        :return: True if the operation was successful, False otherwise.
        """
        return self.SQLITE_HANDLER.update_config(config.model_dump(), config_name)

    def reset_config_db(self, config_name: str) -> bool | None:
        """
        Resets one configuration in the database by re-initializing it with the
        configurations from the YAML template.

        :param config_name: Name of the configuration to reset.
        """
        config = self.CONFIG_TEMPLATE[config_name]
        self.write_config_to_db(config_name, ConfigModelDB(**config))

        return True

    def build_config(self) -> Dict[str, ConfigModelBuild]:
        """
        Builds configurations by substituting user prompts into system prompt templates.

        :return: Dictionary mapping configuration names to their built configurations.
        """
        configs_db = self.load_all_configs_from_db()
        configs = {}
        for config_name, config_db in configs_db.items():
            config = {}
            prompt_template = Template(config_db.model_dump()[self.PROMPT_TEMPLATE])
            user_part = config_db.model_dump()[self.USER_PROMPT]
            config[self.SYSTEM_PROMPT] = prompt_template.substitute(user_part=user_part)
            configs[config_name] = ConfigModelBuild(**config)
        return configs
