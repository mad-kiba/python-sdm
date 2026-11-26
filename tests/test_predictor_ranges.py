"""
Статическая проверка правдоподобия диапазонов (п. 34 ТЗ 1.3, пункт "г").

Это не замена проверке на реальных данных (scripts/check_layer_scales.py читает
реальный стек растров и сравнивает с этими же диапазонами; tests/test_predictors_info.py
фиксирует несколько уже сверенных сырых значений с сервера как регрессионный тест) -
та проверка слишком тяжёлая для обычного прогона тестов (нужен реальный стек растров).

Здесь проверяется только структура справочника predictors_info.py, без растровых данных:
- у каждого непрерывного слоя есть диапазон правдоподобия в EXPECTED (scripts/check_layer_scales.py);
- заявленные в самом справочнике min_val/max_val не противоречат этому диапазону.
"""

import importlib.util
import os

from sdm.utils.predictors_info import get_predictors_info

_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'check_layer_scales.py',
)
_spec = importlib.util.spec_from_file_location('check_layer_scales', _SCRIPT_PATH)
check_layer_scales = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_layer_scales)

INFO = get_predictors_info()


def test_every_continuous_layer_has_a_plausibility_range():
    missing = [
        key for key, info in INFO.items()
        if info.get('kind') != 'categorical' and check_layer_scales.expected_range(key) is None
    ]
    assert not missing, f"Нет диапазона правдоподобия в scripts/check_layer_scales.py EXPECTED для: {missing}"


def test_declared_bounds_do_not_contradict_expected_range():
    for key, info in INFO.items():
        if info.get('kind') == 'categorical':
            continue
        expected = check_layer_scales.expected_range(key)
        if expected is None:
            continue
        exp_low, exp_high = expected
        if 'min_val' in info:
            assert info['min_val'] <= exp_high, (
                f"{key}: min_val={info['min_val']} выше правдоподобного максимума {exp_high}"
            )
        if 'max_val' in info:
            assert info['max_val'] >= exp_low, (
                f"{key}: max_val={info['max_val']} ниже правдоподобного минимума {exp_low}"
            )
