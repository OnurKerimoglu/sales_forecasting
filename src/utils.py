import yaml


class Utils:
    def read_config_for_env(config_path="config.yml", env="local"):
        with open(config_path) as f:
            conf_all = yaml.safe_load(f)
        conf_env = conf_all[env]
        return conf_env


# # for testing:
# if __name__ == "__main__":
#     conf = Utils.read_config_for_env()
#     print(conf)
