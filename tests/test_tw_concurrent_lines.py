import torch
import random
import pytest
from db_tsw.db_tsw import TWConcurrentLines

tw = TWConcurrentLines(mlp=None, device='cuda')

@pytest.mark.parametrize('n_trees', [random.randint(1, 10) for _ in range(5)])
@pytest.mark.parametrize('n_lines', [random.randint(1, 20) for _ in range(5)])
@pytest.mark.parametrize('N', [random.randint(1, 20) for _ in range(5)])
def test_edge_length(n_trees, n_lines, N):
    mass_X = torch.rand(n_trees, n_lines, 2 * N)
    mass_Y = torch.rand(n_trees, n_lines, 2 * N)
    combined_axis_coordinate = torch.randn(n_trees, n_lines, 2 * N)
    mass_X = mass_X.to("cuda")
    mass_Y = mass_Y.to("cuda")
    combined_axis_coordinate = combined_axis_coordinate.to("cuda")
    _, _, edge_length = tw.tw_concurrent_lines(mass_X, mass_Y, combined_axis_coordinate)
    coord = torch.cat((torch.zeros(n_trees, n_lines, 1, device="cuda"), combined_axis_coordinate), dim=2)
    coord = torch.sort(coord, dim=2)[0]
    actual_edge = coord[:, :, 1:] - coord[:, :, :-1]
    actual_edge = actual_edge.to("cuda")
    assert torch.allclose(edge_length, actual_edge) 


# =============================================================================
#                                   Test mass cumsum
# =============================================================================

# test case 1
combined_axis_coordinate_test1 = torch.tensor([[[-1.], [1.]]])
mass_X_test1 = torch.tensor([[[0.1], [0.2]]])
mass_Y_test1 = torch.tensor([[[0.5], [0.6]]])
expected_result_test1 = torch.tensor([[[-0.4], [-0.4]]])

# test case 2
combined_axis_coordinate_test2 = torch.tensor([[[-1., 1.], [1., -1.]]])
mass_X_test2 = torch.tensor([[[0.1, 0.2], [0.1, 0.2]]])
mass_Y_test2 = torch.tensor([[[0.5, 0.6], [0.5, 0.6]]])
expected_mass_X_cumsum_test2 = torch.tensor([[[0.1, 0.2], [0.2, 0.1]]])
expected_mass_Y_cumsum_test2 = torch.tensor([[[0.5, 0.6], [0.6, 0.5]]])
expected_result_test2 = expected_mass_X_cumsum_test2 - expected_mass_Y_cumsum_test2

# test case 3
combined_axis_coordinate_test3 = torch.tensor([
    [[2, -1, -2, 1], [-1, 1, -2, 2]],
    [[2, 3, 1, -2], [-2, 1, 3, 2]]
], dtype=torch.float32)
mass_X_test3 = torch.tensor([
    [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
    [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]
])
expected_mass_X_sorted_test3 = torch.tensor([
    [[0.3, 0.2, 0.4, 0.1], [0.3, 0.1, 0.2, 0.4]],
    [[0.4, 0.3, 0.1, 0.2], [0.1, 0.2, 0.4, 0.3]]
])
expected_mass_X_cumsum_test3 = torch.tensor([
    [[0.3, 0.5, 0.5, 0.1], [0.3, 0.4, 0.6, 0.4]],
    [[0.4, 0.6, 0.3, 0.2], [0.1, 0.9, 0.7, 0.3]]
])
mass_Y_test3 = torch.tensor([
    [[0.5, 0.6, 0.7, 0.8], [0.5, 0.6, 0.7, 0.8]],
    [[0.5, 0.6, 0.7, 0.8], [0.5, 0.6, 0.7, 0.8]],
])
expected_mass_Y_sorted_test3 = torch.tensor([
    [[0.7, 0.6, 0.8, 0.5], [0.7, 0.5, 0.6, 0.8]],
    [[0.8, 0.7, 0.5, 0.6], [0.5, 0.6, 0.8, 0.7]]
])
expected_mass_Y_cumsum_test3 = torch.tensor([
    [[0.7, 1.3, 1.3, 0.5], [0.7, 1.2, 1.4, 0.8]],
    [[0.8, 1.8, 1.1, 0.6], [0.5, 2.1, 1.5, 0.7]]
])
expected_result_test3 = expected_mass_X_cumsum_test3 - expected_mass_Y_cumsum_test3

@pytest.mark.parametrize("mass_X, mass_Y, combined_axis_coordinate, expected_result", [
    (mass_X_test1, mass_Y_test1, combined_axis_coordinate_test1, expected_result_test1),
    (mass_X_test2, mass_Y_test2, combined_axis_coordinate_test2, expected_result_test2),
    (mass_X_test3, mass_Y_test3, combined_axis_coordinate_test3, expected_result_test3)
])
def test_sub_mass_target_cumsum(mass_X, mass_Y, combined_axis_coordinate, expected_result):
    mass_X = mass_X.to("cuda")
    mass_Y = mass_Y.to("cuda")
    combined_axis_coordinate = combined_axis_coordinate.to("cuda")
    expected_result = expected_result.to("cuda")
    _, actual_result, _ = tw.tw_concurrent_lines(mass_X, mass_Y, combined_axis_coordinate)
    assert torch.allclose(actual_result, expected_result)
