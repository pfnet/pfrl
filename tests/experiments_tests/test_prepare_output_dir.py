import contextlib
import itertools
import json
import os
import subprocess
import sys
import tempfile

import pytest

import pfrl


@contextlib.contextmanager
def work_dir(dirname):
    orig_dir = os.getcwd()
    os.chdir(dirname)
    yield
    os.chdir(orig_dir)


def test_is_under_git_control():

    with tempfile.TemporaryDirectory() as tmp:

        # Not under git control
        with work_dir(tmp):
            assert not pfrl.experiments.is_under_git_control()

        # Run: git init
        with work_dir(tmp):
            subprocess.call(["git", "init"])

        # Under git control
        with work_dir(tmp):
            assert pfrl.experiments.is_under_git_control()


def test_generate_exp_id():

    with tempfile.TemporaryDirectory() as tmp:
        with work_dir(tmp):
            subprocess.check_output(["git", "init"])
            subprocess.check_output(["touch", "a"])
            subprocess.check_output(["git", "add", "a"])
            subprocess.check_output(["git", "commit", "-m", "a"])

            id_a = pfrl.experiments.generate_exp_id()
            assert id_a == pfrl.experiments.generate_exp_id()

            assert id_a != pfrl.experiments.generate_exp_id("prefix")

    with tempfile.TemporaryDirectory() as tmp2:
        with work_dir(tmp2):
            subprocess.check_output(["git", "init"])
            subprocess.check_output(["touch", "b"])
            subprocess.check_output(["git", "add", "b"])
            subprocess.check_output(["git", "commit", "-m", "b"])

            id_b = pfrl.experiments.generate_exp_id()
            assert id_a != id_b


@pytest.mark.parametrize(
    "exp_id,git,basedir,argv",
    itertools.product(
        ("my_exp_1", None),
        (True, False),
        ("temp", None),
        (["command", "--option"], None),
    ),
)
def test_prepare_output_dir(exp_id, git, basedir, argv):

    with tempfile.TemporaryDirectory() as tmp:
        if not exp_id and not git:
            pytest.skip("Without git it cannot generate experiment id")

        args = dict(a=1, b="2")
        os.environ["PFRL_TEST_PREPARE_OUTPUT_DIR"] = "T"

        with work_dir(tmp):

            if git:
                subprocess.call(["git", "init"])
                with open("not_utf-8.txt", "wb") as f:
                    f.write(b"\x80")
                subprocess.call(["git", "add", "not_utf-8.txt"])
                subprocess.call(["git", "commit", "-mcommit1"])
                with open("not_utf-8.txt", "wb") as f:
                    f.write(b"\x81")

            dirname = pfrl.experiments.prepare_output_dir(args, basedir, exp_id, argv)

            assert os.path.isdir(dirname)

            if basedir:
                dirname.startswith(basedir)

            # args.txt
            args_path = os.path.join(dirname, "args.txt")
            with open(args_path, "r") as f:
                obj = json.load(f)
                assert obj == args

            # environ.txt
            environ_path = os.path.join(dirname, "environ.txt")
            with open(environ_path, "r") as f:
                obj = json.load(f)
                assert "T" == obj["PFRL_TEST_PREPARE_OUTPUT_DIR"]

            # command.txt
            command_path = os.path.join(dirname, "command.txt")
            with open(command_path, "r") as f:
                if argv:
                    assert " ".join(argv) == f.read()
                else:
                    assert " ".join(sys.argv) == f.read()

            for gitfile in [
                "git-head.txt",
                "git-status.txt",
                "git-log.txt",
                "git-diff.txt",
            ]:
                if git:
                    assert os.path.exists(os.path.join(dirname, gitfile))
                else:
                    assert not os.path.exists(os.path.join(dirname, gitfile))
