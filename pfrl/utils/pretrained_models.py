def download_model(alg, env, model_type="best"):
    """Downloads and returns pretrained model.

    Args:
        alg (string): URL to download from.
        env (string): Gym Environment name.
        model_type (string): Either `best` or `final`.
    Returns:
        str: Path to the downloaded file.
        bool: whether the model was already cached.
    """
    raise NotImplementedError()
