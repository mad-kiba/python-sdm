# sdm/utils/elapid_core/
#
# Урезанный, адаптированный под Python 3.9 перенос "ядра" реализации MaxEnt
# из библиотеки elapid (https://github.com/earth-chris/elapid), без геопространственной
# обвязки (растры, векторы, S3), которая самой модели не нужна.
#
# Исходный код: Copyright (c) 2021 Christopher Anderson, лицензия MIT.
# Полный текст лицензии - в THIRD_PARTY_LICENSES.md в корне проекта.
#
# Сознательные отличия от оригинала (см. комментарии в файлах):
# - убран путь через glmnet (у нас его нет и не будет - используется sklearn-путь,
#   который в elapid и так используется по умолчанию при отсутствии glmnet);
# - вместе с glmnet-путём убраны compute_regularization/compute_lambdas и параметры
#   beta_lqp/beta_hinge/beta_threshold/beta_categorical - в sklearn-пути elapid они
#   не используются вообще (регулировка идёт только через beta_multiplier);
# - убраны NicheEnvelopeModel, EnsembleModel и методы с графиками (не нужны для
#   нашего пайплайна, тянут за собой matplotlib.pyplot без необходимости);
# - typing: `X | Y` (PEP 604, Python 3.10+) заменено на `typing.Union[X, Y]" -
#   на сервере Python 3.9;
# - явно передаётся penalty='l1' в sklearn.LogisticRegression - без этого параметр
#   l1_ratio=1.0 даёт верную L1-регуляризацию только на sklearn>=1.8, а на более
#   старых версиях тихо (без ошибки) откатывается на обычную L2 (проверено эмпирически).
