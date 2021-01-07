import argparse
import datetime
import json
import os
import pickle
import shutil
import subprocess
import sys
from binascii import crc32

import pfrl


def is_under_git_control():
    """Return true iff the current directory is under git control."""
    return pfrl.utils.is_return_code_zero(["git", "rev-parse"])


def generate_exp_id(prefix=None, argv=sys.argv) -> str:
    """Generate reproducible, unique and deterministic experiment id

    The generated id will be string generated from prefix, Git
    checksum, git diff from HEAD and command line arguments.

    Returns:
        A generated experiment id in string (str) which if avialable
        for directory name

    """

    if not is_under_git_control():
        raise RuntimeError("Cannot generate experiment id due to Git lacking.")

    names = []
    if prefix is not None:
        names.append(prefix)

    head = subprocess.check_output("git rev-parse HEAD".split()).strip()
    names.append(head.decode())

    # Caveat: does not work with new files that are not yet cached
    sources = [subprocess.check_output("git diff HEAD".split()), pickle.dumps(argv)]

    for source in sources:
        names.append("%08x" % crc32(source))

    return "-".join(names)


def save_git_information(outdir):
    # Save `git rev-parse HEAD` (SHA of the current commit)
    with open(os.path.join(outdir, "git-head.txt"), "wb") as f:
        f.write(subprocess.check_output("git rev-parse HEAD".split()))

    # Save `git status`
    with open(os.path.join(outdir, "git-status.txt"), "wb") as f:
        f.write(subprocess.check_output("git status".split()))

    # Save `git log`
    with open(os.path.join(outdir, "git-log.txt"), "wb") as f:
        f.write(subprocess.check_output("git log".split()))

    # Save `git diff`
    with open(os.path.join(outdir, "git-diff.txt"), "wb") as f:
        f.write(subprocess.check_output("git diff HEAD".split()))


def prepare_output_dir(
    args,
    basedir=None,
    exp_id=None,
    argv=None,
    time_format="%Y%m%dT%H%M%S.%f",
    make_backup=True,
) -> str:
    """Prepare a directory for outputting training results.

    An output directory, which ends with the current datetime string,
    is created. Then the following infomation is saved into the directory:

        args.txt: argument values and arbitrary parameters
        command.txt: command itself
        environ.txt: environmental variables
        start.txt: timestamp when the experiment executed

    Additionally, if the current directory is under git control, the following
    information is saved:

        git-head.txt: result of `git rev-parse HEAD`
        git-status.txt: result of `git status`
        git-log.txt: result of `git log`
        git-diff.txt: result of `git diff HEAD`

    Args:
        exp_id (str or None): Experiment identifier. If ``None`` is given,
            reproducible ID will be automatically generated from Git version
            hash and command arguments. If the code is not under Git control,
            it is generated from current timestamp under the format of
            ``time_format``.
        args (dict or argparse.Namespace): Arguments to save to see parameters
        basedir (str or None): If a string is specified, the output
            directory is created under that path. If not specified, it is
            created in current directory.
        argv (list or None): The list of command line arguments passed to a
            script. If not specified, sys.argv is used instead.
        time_format (str): Format used to represent the current datetime. The
            default format is the basic format of ISO 8601.
        make_backup (bool): If there exists old experiment with same name,
            copy a backup with additional suffix with ``time_format``.
    Returns:
        Path of the output directory created by this function (str).
    """

    timestamp = datetime.datetime.now().strftime(time_format)

    if exp_id is None:
        if is_under_git_control():
            exp_id = generate_exp_id()
        else:
            exp_id = timestamp

    outdir = os.path.join(basedir or ".", exp_id)

    # Make backup if there's existing output directory. It is
    # recommended for applications not to overwrite files, and try as
    # much as possible to resume or append to existing files. But to
    # prevent unintentional overwrite, the library also makes a backup
    # of the outfile.
    if os.path.exists(outdir) and make_backup:
        backup_dir = "{}.{}.backup".format(outdir, timestamp)
        shutil.copytree(outdir, backup_dir)

    os.makedirs(outdir, exist_ok=True)

    # Save timestamp when the experiment was (re)started
    with open(os.path.join(outdir, "start.txt"), "a") as f:
        # Timestamp created above is not to be reused, because (1)
        # recursive backup of existing outdir may take a long time,
        # and (2) the format of the timestamp must be unified.
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f")
        f.write("{}\n".format(timestamp))

    # Save all the arguments
    with open(os.path.join(outdir, "args.txt"), "w") as f:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        f.write(json.dumps(args))

    # Save all the environment variables
    with open(os.path.join(outdir, "environ.txt"), "w") as f:
        f.write(json.dumps(dict(os.environ)))

    # Save the command
    with open(os.path.join(outdir, "command.txt"), "w") as f:
        if argv is None:
            argv = sys.argv
        f.write(" ".join(argv))

    if is_under_git_control():
        save_git_information(outdir)

    return outdir
