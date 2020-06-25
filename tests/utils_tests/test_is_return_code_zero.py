import unittest

import pfrl


class TestIsReturnCodeZero(unittest.TestCase):
    def test(self):
        # Assume ls command exists
        self.assertTrue(pfrl.utils.is_return_code_zero(["ls"]))
        self.assertFalse(pfrl.utils.is_return_code_zero(["ls --nonexistentoption"]))
        self.assertFalse(pfrl.utils.is_return_code_zero(["nonexistentcommand"]))
