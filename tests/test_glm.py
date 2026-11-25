import numpy as np
import pytest

from sdm.utils.models import GLM


def make_data(rng, n_pres=150, n_bg=600, n_features=4):
    # Унимодальный отклик по первому признаку (оптимум в середине диапазона) - то, что линейная
    # модель без квадратичного члена не может описать, а GLM с ним - может.
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
    model = GLM(random_state=0).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unimodal_response_is_captured_with_quadratic_features(data):
    # Без квадратичного члена логистическая регрессия не может выделить "середину диапазона";
    # это прямая проверка того, зачем add_quadratic нужен методологически, а не только для симметрии с MaxEnt.
    X, y = data
    rng = np.random.default_rng(2)
    X_test, y_test = make_data(rng, n_pres=80, n_bg=320)

    model_quad = GLM(add_quadratic=True, random_state=0).fit(X, y)
    model_linear = GLM(add_quadratic=False, random_state=0).fit(X, y)

    from sklearn.metrics import roc_auc_score
    auc_quad = roc_auc_score(y_test, model_quad.predict_proba(X_test)[:, 1])
    auc_linear = roc_auc_score(y_test, model_linear.predict_proba(X_test)[:, 1])
    assert auc_quad > auc_linear + 0.05


def test_feature_importances_shape_and_normalization(data):
    X, y = data
    model = GLM(random_state=0).fit(X, y)
    importances = model.feature_importances_
    assert importances.shape == (X.shape[1],)  # по числу исходных признаков, не расширенных
    assert np.isclose(importances.sum(), 1.0)
    assert (importances >= 0).all()


def test_selected_c_is_positive_and_reported(data):
    X, y = data
    model = GLM(Cs=5, cv=3, random_state=0).fit(X, y)
    assert model.selected_C_ > 0
    assert model.cv == 3


def test_class_weight_balanced_by_default(data):
    X, y = data
    assert GLM().class_weight == 'balanced'


def test_methods_before_fit_raise():
    model = GLM()
    with pytest.raises(RuntimeError):
        _ = model.feature_importances_
    with pytest.raises(RuntimeError):
        model.predict_proba(np.zeros((3, 4)))


def test_predict_proba_rejects_wrong_feature_count(data):
    X, y = data
    model = GLM(random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        model.predict_proba(np.zeros((5, X.shape[1] + 1)))


def test_glm_not_in_time_projection_models():
    from sdm.sdm import TIME_PROJECTION_MODELS
    assert 'GLM' not in TIME_PROJECTION_MODELS
