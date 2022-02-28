# import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
# dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs", config_name="config")
def main(hydra_config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    import genie.utils.general as utils
    from genie.evaluate import evaluate
    from genie.evaluate_from_file import evaluate_from_file
    from genie.train import train

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    mode = hydra_config["mode"]
    if mode != "evaluate_from_file":
        utils.extras(hydra_config)

    # Pretty print the config files
    if hydra_config.get("print_config"):
        utils.print_config(hydra_config, resolve=True)

    if mode == "train":
        return train(hydra_config)
    elif mode == "evaluate":
        return evaluate(hydra_config)
    elif mode == "evaluate_from_file":
        return evaluate_from_file(hydra_config)
    else:
        raise Exception(f"ERROR: Unexpected running mode passed: {mode}")


if __name__ == "__main__":
    main()
