import pickle
import pytest
from copy import deepcopy
import sys
from pathlib import Path
import numpy as np
import os
from dataclasses import is_dataclass
assignment_name = "assignment2"

this_file = Path(__file__)
tests_folder = this_file.parent
test_data_file = tests_folder.joinpath('test_data.pickle')
project_folder = tests_folder.parent
code_folder = project_folder.joinpath(assignment_name)

sys.path.insert(0, str(code_folder))

import solution  # nopep8
import task2  # nopep8


@pytest.fixture
def test_data():
    with open(test_data_file, 'rb') as file:
        test_data = pickle.load(file)
    return test_data


def compare(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.allclose(a, b)
    elif is_dataclass(a) or is_dataclass(b):
        return str(a) == str(b)
    else:
        return a == b


class TestOutput:

    def test_output__condition_mean(self, test_data):
        for finput in test_data["task2.condition_mean"]:
            params = tuple(finput.values())

            x_1, P_1, z_1, R_1, H_1 = deepcopy(params)

            x_2, P_2, z_2, R_2, H_2 = deepcopy(params)

            cond_mean_1 = task2.condition_mean(x_1, P_1, z_1, R_1, H_1)

            cond_mean_2 = solution.task2.condition_mean(
                x_2, P_2, z_2, R_2, H_2)

            assert compare(cond_mean_1, cond_mean_2)

            assert compare(x_1, x_2)
            assert compare(P_1, P_2)
            assert compare(z_1, z_2)
            assert compare(R_1, R_2)
            assert compare(H_1, H_2)

    def test_output__condition_cov(self, test_data):
        for finput in test_data["task2.condition_cov"]:
            params = tuple(finput.values())

            P_1, R_1, H_1 = deepcopy(params)

            P_2, R_2, H_2 = deepcopy(params)

            conditional_cov_1 = task2.condition_cov(P_1, R_1, H_1)

            conditional_cov_2 = solution.task2.condition_cov(P_2, R_2, H_2)

            assert compare(conditional_cov_1, conditional_cov_2)

            assert compare(P_1, P_2)
            assert compare(R_1, R_2)
            assert compare(H_1, H_2)

    def test_output__get_task_2f(self, test_data):
        for finput in test_data["task2.get_task_2f"]:
            params = tuple(finput.values())

            x_bar_1, P_1, z_c_1, R_c_1, H_c_1, z_r_1, R_r_1, H_r_1 = deepcopy(
                params)

            x_bar_2, P_2, z_c_2, R_c_2, H_c_2, z_r_2, R_r_2, H_r_2 = deepcopy(
                params)

            x_bar_c_1, P_c_1, x_bar_r_1, P_r_1 = task2.get_task_2f(
                x_bar_1, P_1, z_c_1, R_c_1, H_c_1, z_r_1, R_r_1, H_r_1)

            x_bar_c_2, P_c_2, x_bar_r_2, P_r_2 = solution.task2.get_task_2f(
                x_bar_2, P_2, z_c_2, R_c_2, H_c_2, z_r_2, R_r_2, H_r_2)

            assert compare(x_bar_c_1, x_bar_c_2)
            assert compare(P_c_1, P_c_2)
            assert compare(x_bar_r_1, x_bar_r_2)
            assert compare(P_r_1, P_r_2)

            assert compare(x_bar_1, x_bar_2)
            assert compare(P_1, P_2)
            assert compare(z_c_1, z_c_2)
            assert compare(R_c_1, R_c_2)
            assert compare(H_c_1, H_c_2)
            assert compare(z_r_1, z_r_2)
            assert compare(R_r_1, R_r_2)
            assert compare(H_r_1, H_r_2)

    def test_output__get_task_2g(self, test_data):
        for finput in test_data["task2.get_task_2g"]:
            params = tuple(finput.values())

            x_bar_c_1, P_c_1, x_bar_r_1, P_r_1, z_c_1, R_c_1, H_c_1, z_r_1, R_r_1, H_r_1 = deepcopy(
                params)

            x_bar_c_2, P_c_2, x_bar_r_2, P_r_2, z_c_2, R_c_2, H_c_2, z_r_2, R_r_2, H_r_2 = deepcopy(
                params)

            x_bar_cr_1, P_cr_1, x_bar_rc_1, P_rc_1 = task2.get_task_2g(
                x_bar_c_1, P_c_1, x_bar_r_1, P_r_1, z_c_1, R_c_1, H_c_1, z_r_1, R_r_1, H_r_1)

            x_bar_cr_2, P_cr_2, x_bar_rc_2, P_rc_2 = solution.task2.get_task_2g(
                x_bar_c_2, P_c_2, x_bar_r_2, P_r_2, z_c_2, R_c_2, H_c_2, z_r_2, R_r_2, H_r_2)

            assert compare(x_bar_cr_1, x_bar_cr_2)
            assert compare(P_cr_1, P_cr_2)
            assert compare(x_bar_rc_1, x_bar_rc_2)
            assert compare(P_rc_1, P_rc_2)

            assert compare(x_bar_c_1, x_bar_c_2)
            assert compare(P_c_1, P_c_2)
            assert compare(x_bar_r_1, x_bar_r_2)
            assert compare(P_r_1, P_r_2)
            assert compare(z_c_1, z_c_2)
            assert compare(R_c_1, R_c_2)
            assert compare(H_c_1, H_c_2)
            assert compare(z_r_1, z_r_2)
            assert compare(R_r_1, R_r_2)
            assert compare(H_r_1, H_r_2)

    def test_output__get_task_2h(self, test_data):
        for finput in test_data["task2.get_task_2h"]:
            params = tuple(finput.values())

            x_bar_rc_1, P_rc_1 = deepcopy(params)

            x_bar_rc_2, P_rc_2 = deepcopy(params)

            prob_above_line_1 = task2.get_task_2h(x_bar_rc_1, P_rc_1)

            prob_above_line_2 = solution.task2.get_task_2h(x_bar_rc_2, P_rc_2)

            assert compare(prob_above_line_1, prob_above_line_2)

            assert compare(x_bar_rc_1, x_bar_rc_2)
            assert compare(P_rc_1, P_rc_2)


class TestSolutionUsage:

    def test_solution_usage__condition_mean(self, test_data):
        for finput in test_data["task2.condition_mean"][:1]:
            params = finput

            solution.used["task2.condition_mean"] = False

            task2.condition_mean(**params)

            assert not solution.used["task2.condition_mean"], (
                "The function uses the solution")

    def test_solution_usage__condition_cov(self, test_data):
        for finput in test_data["task2.condition_cov"][:1]:
            params = finput

            solution.used["task2.condition_cov"] = False

            task2.condition_cov(**params)

            assert not solution.used["task2.condition_cov"], (
                "The function uses the solution")

    def test_solution_usage__get_task_2f(self, test_data):
        for finput in test_data["task2.get_task_2f"][:1]:
            params = finput

            solution.used["task2.get_task_2f"] = False

            task2.get_task_2f(**params)

            assert not solution.used["task2.get_task_2f"], (
                "The function uses the solution")

    def test_solution_usage__get_task_2g(self, test_data):
        for finput in test_data["task2.get_task_2g"][:1]:
            params = finput

            solution.used["task2.get_task_2g"] = False

            task2.get_task_2g(**params)

            assert not solution.used["task2.get_task_2g"], (
                "The function uses the solution")

    def test_solution_usage__get_task_2h(self, test_data):
        for finput in test_data["task2.get_task_2h"][:1]:
            params = finput

            solution.used["task2.get_task_2h"] = False

            task2.get_task_2h(**params)

            assert not solution.used["task2.get_task_2h"], (
                "The function uses the solution")


if __name__ == '__main__':
    os.environ['_PYTEST_RAISE'] = "1"
    pytest.main()
