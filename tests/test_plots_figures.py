import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdm.utils.plots import plot_roc_auc_curve


def test_plot_roc_auc_curve_does_not_leak_figures(tmp_path):
    # Без plt.close() фигура остаётся открытой до конца жизни процесса (долгоживущий сервер
    # накапливает их через множество запросов - именно так проявляется "More than 20 figures").
    open_before = len(plt.get_fignums())
    fpr, tpr = np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.8, 1.0])
    plot_roc_auc_curve(fpr, tpr, 0.85, os.path.join(tmp_path, "roc.jpg"), 1)
    assert len(plt.get_fignums()) == open_before


def test_plot_roc_auc_curve_many_calls_do_not_accumulate(tmp_path):
    fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    for i in range(25):
        plot_roc_auc_curve(fpr, tpr, 0.5, os.path.join(tmp_path, f"roc_{i}.jpg"), 1)
    assert len(plt.get_fignums()) == 0
