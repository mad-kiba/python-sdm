# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sdm", # Название пакета, которое будет использоваться при pip install
    version="0.9", # должна совпадать с sdm/__init__.py: __version__
    author="madkiba",
    author_email="ogletix@gmail.com",
    description="Python library for Species Distribution Modeling (Python-SDM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mad-kiba/python-sdm",
    packages=find_packages(exclude=["tests", "examples"]), # Автоматически находит пакеты
    install_requires=[
        # Список сверен с реальными импортами в sdm/ (geopandas и seaborn раньше
        # были здесь, но нигде в коде не использовались - убраны).
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.7.0",
        "rasterio>=1.2.0",
        "affine>=2.3.0",
        "pyproj>=3.0.0",
        "contextily>=1.2.0",
        "matplotlib>=3.5.0",
        "opencv-python>=4.5.0",
        "Pillow>=9.0.0",
        "joblib>=1.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: GIS",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.9", # минимум, реально используемый (сервер - 3.9); совместимость не проверялась ниже этой версии
    #entry_points={
    #    # Если вы делаете CLI утилиту
    #    'console_scripts': [
    #        'sdm=sdm.cli.sdm_cli:main',
    #    ],
    #},
)