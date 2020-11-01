import os
import tempfile
import unittest

import torch

import pfrl


def create_simple_link():
    link = torch.nn.Module()
    link.param = torch.nn.Parameter(torch.zeros(1))
    return link


class Parent(pfrl.agent.AttributeSavingMixin, object):

    saved_attributes = ("link", "child")

    def __init__(self):
        self.link = create_simple_link()
        self.child = Child()


class Child(pfrl.agent.AttributeSavingMixin, object):

    saved_attributes = ("link",)

    def __init__(self):
        self.link = create_simple_link()


class Parent2(pfrl.agent.AttributeSavingMixin, object):

    saved_attributes = ("child_a", "child_b")

    def __init__(self, child_a, child_b):
        self.child_a = child_a
        self.child_b = child_b


class TestAttributeSavingMixin(unittest.TestCase):
    def test_save_load(self):
        parent = Parent()
        parent.link.param.detach().numpy()[:] = 1
        parent.child.link.param.detach().numpy()[:] = 2
        # Save
        dirname = tempfile.mkdtemp()
        parent.save(dirname)
        self.assertTrue(os.path.isdir(dirname))
        self.assertTrue(os.path.isfile(os.path.join(dirname, "link.pt")))
        self.assertTrue(os.path.isdir(os.path.join(dirname, "child")))
        self.assertTrue(os.path.isfile(os.path.join(dirname, "child", "link.pt")))
        # Load
        parent = Parent()
        self.assertEqual(int(parent.link.param.detach().numpy()), 0)
        self.assertEqual(int(parent.child.link.param.detach().numpy()), 0)
        parent.load(dirname)
        self.assertEqual(int(parent.link.param.detach().numpy()), 1)
        self.assertEqual(int(parent.child.link.param.detach().numpy()), 2)

    def test_save_load_2(self):
        parent = Parent()
        parent2 = Parent2(parent.child, parent)
        # Save
        dirname = tempfile.mkdtemp()
        parent2.save(dirname)
        # Load
        parent = Parent()
        parent2 = Parent2(parent.child, parent)
        parent2.load(dirname)

    def test_loop1(self):
        parent = Parent()
        parent.child = parent
        dirname = tempfile.mkdtemp()

        # The assertion in PFRL should fail on save().
        # Otherwise it seems to raise OSError: [Errno 63] File name too long
        with self.assertRaises(AssertionError):
            parent.save(dirname)

    def test_loop2(self):
        parent1 = Parent()
        parent2 = Parent()
        parent1.child = parent2
        parent2.child = parent1
        dirname = tempfile.mkdtemp()

        # The assertion in PFRL should fail on save().
        # Otherwise it seems to raise OSError: [Errno 63] File name too long
        with self.assertRaises(AssertionError):
            parent1.save(dirname)

    def test_with_data_parallel(self):
        parent = Parent()
        parent.link.param.detach().numpy()[:] = 1
        parent.child.link.param.detach().numpy()[:] = 2
        parent.link = torch.nn.DataParallel(parent.link)

        # Save
        dirname = tempfile.mkdtemp()
        parent.save(dirname)
        self.assertTrue(os.path.isdir(dirname))
        self.assertTrue(os.path.isfile(os.path.join(dirname, "link.pt")))
        self.assertTrue(os.path.isdir(os.path.join(dirname, "child")))
        self.assertTrue(os.path.isfile(os.path.join(dirname, "child", "link.pt")))

        # Load Parent without data parallel
        parent = Parent()
        self.assertEqual(int(parent.link.param.detach().numpy()), 0)
        self.assertEqual(int(parent.child.link.param.detach().numpy()), 0)
        parent.load(dirname)
        self.assertEqual(int(parent.link.param.detach().numpy()), 1)
        self.assertEqual(int(parent.child.link.param.detach().numpy()), 2)

        # Load Parent with data parallel
        parent = Parent()
        parent.link = torch.nn.DataParallel(parent.link)
        self.assertEqual(int(parent.link.module.param.detach().numpy()), 0)
        self.assertEqual(int(parent.child.link.param.detach().numpy()), 0)
        parent.load(dirname)
        self.assertEqual(int(parent.link.module.param.detach().numpy()), 1)
        self.assertEqual(int(parent.child.link.param.detach().numpy()), 2)
