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
import pdaf  # nopep8


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


class Test_PDAF_predict_state:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["pdaf.PDAF.predict_state"]:
            params = tuple(finput.values())

            self_1, state_upd_prev_gauss_1, Ts_1 = deepcopy(params)

            self_2, state_upd_prev_gauss_2, Ts_2 = deepcopy(params)

            state_pred_gauus_1 = pdaf.PDAF.predict_state(
                self_1, state_upd_prev_gauss_1, Ts_1)

            state_pred_gauus_2 = solution.pdaf.PDAF.predict_state(
                self_2, state_upd_prev_gauss_2, Ts_2)

            assert compare(state_pred_gauus_1, state_pred_gauus_2)

            assert compare(self_1, self_2)
            assert compare(state_upd_prev_gauss_1, state_upd_prev_gauss_2)
            assert compare(Ts_1, Ts_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["pdaf.PDAF.predict_state"][:1]:
            params = finput

            solution.used["pdaf.PDAF.predict_state"] = False

            pdaf.PDAF.predict_state(**params)

            assert not solution.used["pdaf.PDAF.predict_state"], "The function uses the solution"


class Test_PDAF_predict_measurement:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["pdaf.PDAF.predict_measurement"]:
            params = tuple(finput.values())

            self_1, state_pred_gauss_1 = deepcopy(params)

            self_2, state_pred_gauss_2 = deepcopy(params)

            z_pred_gauss_1 = pdaf.PDAF.predict_measurement(
                self_1, state_pred_gauss_1)

            z_pred_gauss_2 = solution.pdaf.PDAF.predict_measurement(
                self_2, state_pred_gauss_2)

            assert compare(z_pred_gauss_1, z_pred_gauss_2)

            assert compare(self_1, self_2)
            assert compare(state_pred_gauss_1, state_pred_gauss_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["pdaf.PDAF.predict_measurement"][:1]:
            params = finput

            solution.used["pdaf.PDAF.predict_measurement"] = False

            pdaf.PDAF.predict_measurement(**params)

            assert not solution.used["pdaf.PDAF.predict_measurement"], "The function uses the solution"


class Test_PDAF_gate:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["pdaf.PDAF.gate"]:
            params = tuple(finput.values())

            self_1, z_pred_gauss_1, measurements_1 = deepcopy(params)

            self_2, z_pred_gauss_2, measurements_2 = deepcopy(params)

            gated_measurements_1 = pdaf.PDAF.gate(
                self_1, z_pred_gauss_1, measurements_1)

            gated_measurements_2 = solution.pdaf.PDAF.gate(
                self_2, z_pred_gauss_2, measurements_2)

            assert compare(gated_measurements_1, gated_measurements_2)

            assert compare(self_1, self_2)
            assert compare(z_pred_gauss_1, z_pred_gauss_2)
            assert compare(measurements_1, measurements_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["pdaf.PDAF.gate"][:1]:
            params = finput

            solution.used["pdaf.PDAF.gate"] = False

            pdaf.PDAF.gate(**params)

            assert not solution.used["pdaf.PDAF.gate"], "The function uses the solution"


class Test_PDAF_get_association_prob:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["pdaf.PDAF.get_association_prob"]:
            params = tuple(finput.values())

            self_1, z_pred_gauss_1, gated_measurements_1 = deepcopy(params)

            self_2, z_pred_gauss_2, gated_measurements_2 = deepcopy(params)

            associations_probs_1 = pdaf.PDAF.get_association_prob(
                self_1, z_pred_gauss_1, gated_measurements_1)

            associations_probs_2 = solution.pdaf.PDAF.get_association_prob(
                self_2, z_pred_gauss_2, gated_measurements_2)

            assert compare(associations_probs_1, associations_probs_2)

            assert compare(self_1, self_2)
            assert compare(z_pred_gauss_1, z_pred_gauss_2)
            assert compare(gated_measurements_1, gated_measurements_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["pdaf.PDAF.get_association_prob"][:1]:
            params = finput

            solution.used["pdaf.PDAF.get_association_prob"] = False

            pdaf.PDAF.get_association_prob(**params)

            assert not solution.used["pdaf.PDAF.get_association_prob"], "The function uses the solution"


class Test_PDAF_get_cond_update_gaussians:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["pdaf.PDAF.get_cond_update_gaussians"]:
            params = tuple(finput.values())

            self_1, state_pred_gauss_1, z_pred_gauss_1, gated_measurements_1 = deepcopy(
                params)

            self_2, state_pred_gauss_2, z_pred_gauss_2, gated_measurements_2 = deepcopy(
                params)

            update_gaussians_1 = pdaf.PDAF.get_cond_update_gaussians(
                self_1, state_pred_gauss_1, z_pred_gauss_1, gated_measurements_1)

            update_gaussians_2 = solution.pdaf.PDAF.get_cond_update_gaussians(
                self_2, state_pred_gauss_2, z_pred_gauss_2, gated_measurements_2)

            assert compare(update_gaussians_1, update_gaussians_2)

            assert compare(self_1, self_2)
            assert compare(state_pred_gauss_1, state_pred_gauss_2)
            assert compare(z_pred_gauss_1, z_pred_gauss_2)
            assert compare(gated_measurements_1, gated_measurements_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["pdaf.PDAF.get_cond_update_gaussians"][:1]:
            params = finput

            solution.used["pdaf.PDAF.get_cond_update_gaussians"] = False

            pdaf.PDAF.get_cond_update_gaussians(**params)

            assert not solution.used["pdaf.PDAF.get_cond_update_gaussians"], "The function uses the solution"


class Test_PDAF_update:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["pdaf.PDAF.update"]:
            params = tuple(finput.values())

            self_1, state_pred_gauss_1, z_pred_gauss_1, measurements_1 = deepcopy(
                params)

            self_2, state_pred_gauss_2, z_pred_gauss_2, measurements_2 = deepcopy(
                params)

            state_upd_gauss_1 = pdaf.PDAF.update(
                self_1, state_pred_gauss_1, z_pred_gauss_1, measurements_1)

            state_upd_gauss_2 = solution.pdaf.PDAF.update(
                self_2, state_pred_gauss_2, z_pred_gauss_2, measurements_2)

            assert compare(state_upd_gauss_1, state_upd_gauss_2)

            assert compare(self_1, self_2)
            assert compare(state_pred_gauss_1, state_pred_gauss_2)
            assert compare(z_pred_gauss_1, z_pred_gauss_2)
            assert compare(measurements_1, measurements_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["pdaf.PDAF.update"][:1]:
            params = finput

            solution.used["pdaf.PDAF.update"] = False

            pdaf.PDAF.update(**params)

            assert not solution.used["pdaf.PDAF.update"], "The function uses the solution"


class Test_PDAF_step_with_info:
    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for finput in test_data["pdaf.PDAF.step_with_info"]:
            params = tuple(finput.values())

            self_1, state_upd_prev_gauss_1, measurements_1, Ts_1 = deepcopy(
                params)

            self_2, state_upd_prev_gauss_2, measurements_2, Ts_2 = deepcopy(
                params)

            state_pred_gauss_1, z_pred_gauss_1, state_upd_gauss_1 = pdaf.PDAF.step_with_info(
                self_1, state_upd_prev_gauss_1, measurements_1, Ts_1)

            state_pred_gauss_2, z_pred_gauss_2, state_upd_gauss_2 = solution.pdaf.PDAF.step_with_info(
                self_2, state_upd_prev_gauss_2, measurements_2, Ts_2)

            assert compare(state_pred_gauss_1, state_pred_gauss_2)
            assert compare(z_pred_gauss_1, z_pred_gauss_2)
            assert compare(state_upd_gauss_1, state_upd_gauss_2)

            assert compare(self_1, self_2)
            assert compare(state_upd_prev_gauss_1, state_upd_prev_gauss_2)
            assert compare(measurements_1, measurements_2)
            assert compare(Ts_1, Ts_2)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        for finput in test_data["pdaf.PDAF.step_with_info"][:1]:
            params = finput

            solution.used["pdaf.PDAF.step_with_info"] = False

            pdaf.PDAF.step_with_info(**params)

            assert not solution.used["pdaf.PDAF.step_with_info"], "The function uses the solution"


if __name__ == "__main__":
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
