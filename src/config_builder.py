import yaml
from src.utils import sqlite_db_handler


class ConfigBuilder:
    def __init__(self, config_file_path: str, table_name: str = "configs"):
        self._config_template = self.load_yaml(config_file_path)
        self._sqlite_handler = sqlite_db_handler.SqliteDBHandler(table_name)

    @staticmethod
    def load_yaml(config_path: str) -> dict:
        """
        Loads in a system prompt (and additional parameters) for the LLM and returns them as a tuple.
        :param config_path: path to yaml file
        :return: Tuple containing the system prompt and additional parameters
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def load_config_from_db(self, config_name: str) -> dict:
        config = self._sqlite_handler.get_config(config_name)
        config.pop("system_prompt_template", None)
        return config

    def load_all_configs_from_db(self) -> dict:
        config_names = self._sqlite_handler.get_all_config_names()

        all_configs = {}
        for config_name in config_names:
            all_configs[config_names] = self.load_config_from_db(config_name)

        return all_configs

    def write_config_to_db(self, config_name: str, config: dict) -> bool | None:
        config.pop("system_prompt_template", None)
        return self._sqlite_handler.add_config(config, config_name)

    def delete_config_from_db(self, config_name: str) -> bool | None:
        return self._sqlite_handler.delete_config(config_name)

    def build_config(self) -> dict:
        configs = self._config_template.copy()
        for config_name, config in configs.items():
            loaded_config = self.load_config_from_db(config_name)
            if len(loaded_config) == 0:
                _ = self.write_config_to_db(config_name, config)
            else:
                config.update(loaded_config)

            config["system_prompt"] = config["system_prompt_template"].format(user_part=config["system_prompt_user"])
        return configs