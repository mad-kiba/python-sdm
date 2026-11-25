import numpy as np
import pytest

from sdm.utils.models import MaxEnt


def make_data(rng, n_pres=150, n_bg=600, n_features=4):
    x_bg = rng.normal(0, 2, (n_bg, n_features))
    centers = rng.uniform(-1, 1, n_pres)
    x_pres = rng.normal(0, 2, (n_pres, n_features))
    x_pres[:, 0] = centers + rng.normal(0, 0.3, n_pres)
    X = np.vstack([x_pres, x_bg])
    y = np.concatenate([np.ones(n_pres, dtype=int), np.zeros(n_bg, dtype=int)])
    return X, y


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    return make_data(rng)


def test_fit_predict_proba_shape_and_range(data):
    X, y = data
    model = MaxEnt(random_state=0, n_importance_repeats=3).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_feature_importances_shape_and_normalization(data):
    X, y = data
    model = MaxEnt(random_state=0, n_importance_repeats=3).fit(X, y)
    importances = model.feature_importances_
    assert importances.shape == (X.shape[1],)  # по числу исходных, не расширенных признаков
    assert np.isclose(importances.sum(), 1.0)
    assert (importances >= 0).all()  # отрицательного шума permutation importance не просочилось


def test_informative_feature_ranks_highest(data):
    # В make_data только признак 0 реально разделяет присутствие/фон - должен быть самым важным.
    X, y = data
    model = MaxEnt(random_state=0, n_importance_repeats=5).fit(X, y)
    assert np.argmax(model.feature_importances_) == 0


def test_default_feature_types_match_elapid_default(data):
    X, y = data
    model = MaxEnt(random_state=0).fit(X, y)
    assert list(model.feature_types) == ["linear", "hinge", "product"]


def test_custom_feature_types_and_beta_multiplier_are_passed_through(data):
    X, y = data
    model = MaxEnt(feature_types=["linear"], beta_multiplier=3.0, random_state=0).fit(X, y)
    assert list(model.feature_types) == ["linear"]
    assert model.beta_multiplier == 3.0


def test_methods_before_fit_raise():
    model = MaxEnt()
    with pytest.raises(RuntimeError):
        _ = model.feature_importances_
    with pytest.raises(RuntimeError):
        model.predict_proba(np.zeros((3, 4)))


def test_predict_proba_rejects_wrong_feature_count(data):
    X, y = data
    model = MaxEnt(random_state=0, n_importance_repeats=3).fit(X, y)
    with pytest.raises(ValueError):
        model.predict_proba(np.zeros((5, X.shape[1] + 1)))


def test_maxent_not_in_time_projection_models():
    from sdm.sdm import TIME_PROJECTION_MODELS
    assert 'MaxEnt' not in TIME_PROJECTION_MODELS
