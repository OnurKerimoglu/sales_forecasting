import yaml


class Utils:
    """
    A utility class with helper functions.
    Methods:
        read_config_for_env(config_path, env): Reads the config file for the specified environment.
    """
    @staticmethod
    def read_config_for_env(config_path="config/config.yml", env="local"):
        """
        Reads the config file for the specified environment.
        Args:
            config_path (str): Path of the configuration file.
            env (str): The environment for which to read the config.
        Returns:
            dict: The configuration parameters for the specified environment.
        """
        with open(config_path) as f:
            conf_all = yaml.safe_load(f)
        conf_env = conf_all[env]
        return conf_env


# # for testing:
if __name__ == "__main__":
    conf = Utils.read_config_for_env()
    print(conf)
