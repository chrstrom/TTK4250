import pickle
from numpy.core.numeric import isscalar
import pytest
from copy import deepcopy
import sys
from pathlib import Path
import numpy as np
import os
from dataclasses import is_dataclass

assignment_name = "pda"

this_file = Path(__file__)
tests_folder = this_file.parent
test_data_file = tests_folder.joinpath("test_data.pickle")
project_folder = tests_folder.parent
code_folder = project_folder.joinpath(assignment_name)

sys.path.insert(0, str(code_folder))

import solution  # nopep8
import gaussmix  # nopep8


@pytest.fixture
def test_data():
    with open(test_data_file, "rb") as file:
        test_data = pickle.load(file)
    return test_data


def compare(a, b):
    if (
        isinstance(a, np.ndarray)
        or isinstance(b, np.ndarray)
        or np.isscalar(a)
        or np.isscalar(b)
    ):
        return np.allclose(a, b)
    elif is_dataclass(a) or is_dataclass(b):
        return str(a) == str(b)
    else:
        return a == b


class Test_GaussianMuxture_get_mean:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["gaussmix.GaussianMuxture.get_mean"]:
            params = tuple(finput.values())

            self_1, = deepcopy(params)

            self_2, = deepcopy(params)

            mean_1 = gaussmix.GaussianMuxture.get_mean(self_1,)

            mean_2 = solution.gaussmix.GaussianMuxture.get_mean(self_2,)

            assert compare(mean_1, mean_2)

            assert compare(self_1, self_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["gaussmix.GaussianMuxture.get_mean"][:1]:
            params = finput

            solution.used["gaussmix.GaussianMuxture.get_mean"] = False

            gaussmix.GaussianMuxture.get_mean(**params)

            assert not solution.used["gaussmix.GaussianMuxture.get_mean"], "The function uses the solution"


class Test_GaussianMuxture_get_cov:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["gaussmix.GaussianMuxture.get_cov"]:
            params = tuple(finput.values())

            self_1, = deepcopy(params)

            self_2, = deepcopy(params)

            cov_1 = gaussmix.GaussianMuxture.get_cov(self_1,)

            cov_2 = solution.gaussmix.GaussianMuxture.get_cov(self_2,)

            assert compare(cov_1, cov_2)

            assert compare(self_1, self_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["gaussmix.GaussianMuxture.get_cov"][:1]:
            params = finput

            solution.used["gaussmix.GaussianMuxture.get_cov"] = False

            gaussmix.GaussianMuxture.get_cov(**params)

            assert not solution.used["gaussmix.GaussianMuxture.get_cov"], "The function uses the solution"


class Test_GaussianMuxture_reduce:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["gaussmix.GaussianMuxture.reduce"]:
            params = tuple(finput.values())

            self_1, = deepcopy(params)

            self_2, = deepcopy(params)

            reduction_1 = gaussmix.GaussianMuxture.reduce(self_1,)

            reduction_2 = solution.gaussmix.GaussianMuxture.reduce(self_2,)

            assert compare(reduction_1, reduction_2)

            assert compare(self_1, self_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["gaussmix.GaussianMuxture.reduce"][:1]:
            params = finput

            solution.used["gaussmix.GaussianMuxture.reduce"] = False

            gaussmix.GaussianMuxture.reduce(**params)

            assert not solution.used["gaussmix.GaussianMuxture.reduce"], "The function uses the solution"


if __name__ == "__main__":
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
