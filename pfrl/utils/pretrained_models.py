"""This file is a fork from ChainerCV, an MIT-licensed project,
https://github.com/chainer/chainercv/blob/master/chainercv/utils/download.py
"""

import hashlib
import os
import posixpath
import shutil
import sys
import tempfile
import time
import urllib
import zipfile

import filelock

_models_root = os.environ.get(
    "PFRL_MODELS_ROOT", os.path.join(os.path.expanduser("~"), ".pfrl", "models")
)


MODELS = {
    "DQN": ["best", "final"],
    "IQN": ["best", "final"],
    "Rainbow": ["best", "final"],
    "A3C": ["best", "final"],
    "DDPG": ["best", "final"],
    "TRPO": ["best", "final"],
    "PPO": ["final"],
    "TD3": ["best", "final"],
    "SAC": ["best", "final"],
}

download_url = "https://pfrl-assets.preferred.jp/"


def _get_model_directory(model_name, create_directory=True):
    """Gets the path to the directory of given model.

    The generated path is just a concatenation of the global root directory
    and the model name. This function forked from Chainer, an MIT-licensed project,
    https://github.com/chainer/chainer/blob/v7.4.0/chainer/dataset/download.py#L43
    Args:
        model_name (str): Name of the model.
        create_directory (bool): If True (default), this function also creates
            the directory at the first time. If the directory already exists,
            then this option is ignored.
    Returns:
        str: Path to the dataset directory.
    """
    path = os.path.join(_models_root, model_name)
    if create_directory:
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
    return path


def _reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        print("  %   Total    Recv       Speed  Time left")
        return
    duration = time.time() - start_time
    progress_size = count * block_size
    try:
        speed = progress_size / duration
    except ZeroDivisionError:
        speed = float("inf")
    percent = progress_size / total_size * 100
    eta = int((total_size - progress_size) / speed)
    sys.stdout.write(
        "\r{:3.0f} {:4.0f}MiB {:4.0f}MiB {:6.0f}KiB/s {:4d}:{:02d}:{:02d}".format(
            percent,
            total_size / (1 << 20),
            progress_size / (1 << 20),
            speed / (1 << 10),
            eta // 60 // 60,
            (eta // 60) % 60,
            eta % 60,
        )
    )
    sys.stdout.flush()


def cached_download(url):
    """Downloads a file and caches it.

    It downloads a file from the URL if there is no corresponding cache.
    If there is already a cache for the given URL, it just returns the
    path to the cache without downloading the same file.
    This function forked from Chainer, an MIT-licensed project,
    https://github.com/chainer/chainer/blob/v7.4.0/chainer/dataset/download.py#L70
    Args:
        url (string): URL to download from.
    Returns:
        string: Path to the downloaded file.
    """
    cache_root = os.path.join(_models_root, "_dl_cache")
    try:
        os.makedirs(cache_root)
    except OSError:
        if not os.path.exists(cache_root):
            raise
    lock_path = os.path.join(cache_root, "_dl_lock")
    urlhash = hashlib.md5(url.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_root, urlhash)

    with filelock.FileLock(lock_path):
        if os.path.exists(cache_path):
            return cache_path
    temp_root = tempfile.mkdtemp(dir=cache_root)
    try:
        temp_path = os.path.join(temp_root, "dl")
        print("Downloading ...")
        print("From: {:s}".format(url))
        print("To: {:s}".format(cache_path))
        urllib.request.urlretrieve(url, temp_path, _reporthook)
        with filelock.FileLock(lock_path):
            shutil.move(temp_path, cache_path)
    finally:
        shutil.rmtree(temp_root)

    return cache_path


def download_and_store_model(alg, url, env, model_type):
    """Downloads a model file and puts it under model directory.

    It downloads a file from the URL and puts it under model directory.
    If there is already a file at the destination path,
    it just returns the path without downloading the same file.
    Args:
        alg (string): String representation of algorithm used in MODELS dict.
        url (string): URL to download from.
        env (string): Environment in which pretrained model was trained.
        model_type (string): Either `best` or `final`.
    Returns:
        string: Path to the downloaded file.
        bool: whether the model was already cached.
    """
    lock = os.path.join(_get_model_directory(".lock"), "models.lock")
    with filelock.FileLock(lock):
        root = _get_model_directory(os.path.join(alg, env))
        url_basepath = posixpath.join(url, alg, env)
        file = model_type + ".zip"
        path = os.path.join(root, file)
        is_cached = os.path.exists(path)
        if not is_cached:
            cache_path = cached_download(posixpath.join(url_basepath, file))
            os.rename(cache_path, path)
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(root)
        return os.path.join(root, model_type), is_cached


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
    assert alg in MODELS, "No pretrained models for " + alg + "."
    assert model_type in MODELS[alg], (
        'Model type "' + model_type + '" is not supported.'
    )
    env = env.replace("NoFrameskip-v4", "")
    model_path, is_cached = download_and_store_model(alg, download_url, env, model_type)
    return model_path, is_cached
