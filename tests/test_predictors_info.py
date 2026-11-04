import pytest

from sdm.utils.predictors_info import get_predictors_info
from sdm.utils.utils import inverse_scale

INFO = get_predictors_info()


def test_continuous_layers_have_unit():
    empty = [k for k, v in INFO.items() if v.get('kind') != 'categorical' and not v.get('unit')]
    assert not empty


def test_kind_is_explicit_for_one_hot_layers():
    for key, value in INFO.items():
        is_one_hot = value['unit'] == 'да'
        assert (value.get('kind') == 'categorical') == is_one_hot, key
    assert sum(v.get('kind') == 'categorical' for v in INFO.values()) > 0


def test_old_unit_spellings_are_gone():
    assert not {v['unit'] for v in INFO.values()} & {'дн', 'раз', 'кг/кв.м', 'кг/(кв.м.*мес)', 'кг/(кв.м.*год)'}


def test_mean_and_min_titles_are_not_swapped():
    for var in ('pet', 'rsds', 'sfcWind'):
        assert 'Средн' in INFO[f'CHELSA_{var}mean_1981-2010_V.2.1']['title']
        assert 'Минимальн' in INFO[f'CHELSA_{var}min_1981-2010_V.2.1']['title']


def test_last_growing_day_titles_match_threshold():
    for threshold in (0, 5, 10):
        assert f'выше {threshold} °C' in INFO[f'CHELSA_gddlgd{threshold}_1981-2010_V.2.1']['title']


@pytest.mark.parametrize("band, raw, low, high", [
    # Сырые значения в Алматы из проверки на серверном стеке, границы - физически разумные значения
    ('CHELSA_rsdsmean_1981-2010_V.2.1', 161, 5, 30),
    ('CHELSA_rsdsrange_1981-2010_V.2.1', 167, 5, 35),
    ('CHELSA_sfcWindmax_1981-2010_V.2.1', 121, 0.5, 25),
    ('CHELSA_vpdrange_1981-2010_V.2.1', 12440, 50, 8000),
    ('CHELSA_swb_1981-2010_V.2.1', -40, -2000, 2000),
    ('CHELSA_bio03_1981-2010_V.2.1', 264, 10, 60),
    ('wc2.1_30s_bio_3', 27.49, 10, 60),
    ('wc2.1_30s_bio_4', 1099, 5, 20),
    ('ENVIREM_PETseasonality', 5708, 5, 150),
    ('ENVIREM_growingDegDays0', 30910, 0, 7000),
])
def test_layer_scale_on_real_values(band, raw, low, high):
    # raw кладём в mean: при нулевом стандартизованном значении inverse_scale вернёт (raw - diff) / scale
    value = float(inverse_scale(0.0, {'mean': raw, 'scale': 1.0}, INFO[band]))
    assert low <= value <= high


def test_swb_matches_water_balance():
    # Актау: осадки 178 мм, PET 86.91 мм/мес, поэтому баланс около -865 мм при сыром значении -872
    assert INFO['CHELSA_swb_1981-2010_V.2.1']['scale'] == 1


def test_vpd_scale_is_consistent():
    scales = {INFO[f'CHELSA_vpd{v}_1981-2010_V.2.1']['scale'] for v in ('mean', 'min', 'max', 'range')}
    assert scales == {10}
