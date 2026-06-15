import numpy as np

from lbt.eval.items import SOURCE_SCENARIOS, TARGET_SCENARIOS, build_items
from lbt.interp.directions import diff_of_means, extract_directions, select_layer
from lbt.interp.lora_analysis import amplification_ratio, delta_w, effective_rank
from lbt.interp.projections import project, projection_shift
from lbt.interp.steering import gap_removed_fraction


def test_target_domains_are_six_and_held_out():
    assert len(TARGET_SCENARIOS) == 6
    # target and source domain names must not overlap (held-out vs trained)
    assert set(TARGET_SCENARIOS) & set(SOURCE_SCENARIOS) == set()


def test_build_items_shapes():
    items = build_items(TARGET_SCENARIOS, paraphrases=3, contrast_templates=4)
    # 6 domains x 3 scenarios x 3 paraphrases
    assert len(items) == 6 * 3 * 3
    it = items[0]
    assert len(it["contrasts"]) == 4
    for c in it["contrasts"]:
        assert {"question", "question_swapped", "pro", "anti"} <= set(c)
    assert it["forced_choice"]["pro_label"] == "A"
    # ids unique
    assert len({i["id"] for i in items}) == len(items)


def test_diff_of_means_unit_and_direction():
    plus = np.array([[0.0, 0.0], [0.0, 0.0]])
    minus = np.array([[2.0, 0.0], [2.0, 0.0]])
    unit, norm = diff_of_means(plus, minus)
    assert np.allclose(unit, [1.0, 0.0])
    assert norm == 2.0


def test_extract_and_select_layer_picks_best_probe():
    rng = np.random.default_rng(0)
    # layer 0: separable; layer 1: noise
    plus = {0: rng.normal(-2, 0.2, (40, 4)), 1: rng.normal(0, 1, (40, 4))}
    minus = {0: rng.normal(2, 0.2, (40, 4)), 1: rng.normal(0, 1, (40, 4))}
    dirs = extract_directions(plus, minus, val_frac=0.25, seed=0)
    assert select_layer(dirs) == 0
    assert dirs[0].probe_acc >= dirs[1].probe_acc


def test_projection_shift_paired():
    base = {0: np.array([[0.0, 0.0], [0.0, 0.0]])}
    arm = {0: np.array([[1.0, 0.0], [1.0, 0.0]])}
    dirs = {0: np.array([1.0, 0.0])}
    shift = projection_shift(base, arm, dirs)
    assert shift[0]["mean_shift"] == 1.0
    assert shift[0]["n"] == 2


def test_project_scalar():
    acts = np.array([[3.0, 4.0]])
    assert project(acts, np.array([1.0, 0.0]))[0] == 3.0


def test_gap_removed_fraction():
    # framed=2, neutral=0, ablated back to 0 -> full removal
    assert gap_removed_fraction(2.0, 0.0, 0.0) == 1.0
    # no change after ablation -> 0 removed
    assert gap_removed_fraction(2.0, 0.0, 2.0) == 0.0


def test_lora_delta_and_amplification():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (4, 16))  # [r, in]
    B = rng.normal(0, 1, (8, 4))  # [out, r]
    dW = delta_w(A * 0 + 1.0, B, alpha=32, r=4)  # shape [8,16]
    assert dW.shape == (8, 16)
    er = effective_rank(dW)
    assert 0 < er["effective_rank"] <= 4 + 1e-6
    # amplification along an arbitrary unit direction is finite/positive
    amp = amplification_ratio(dW, rng.normal(0, 1, 16), n_random=8, seed=0)
    assert amp > 0
