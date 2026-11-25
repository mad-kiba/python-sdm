from sdm.sdm import TIME_PROJECTION_MODELS


def test_whitelist_is_exactly_rf_and_xgboost():
    # Белый список, а не "все, кроме MaxEnt": новый алгоритм по умолчанию не должен попадать
    # в прогноз будущего/прошлого, пока для него явно не написана ветка в predict_past.
    assert TIME_PROJECTION_MODELS == ('RandomForest', 'XGBoost')


def test_unknown_future_algorithms_are_excluded_by_default():
    for name in ('MaxEnt', 'GLM', 'GAM', 'SomeFutureModel', ''):
        assert name not in TIME_PROJECTION_MODELS
