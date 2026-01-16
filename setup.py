# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sdm", # Название пакета, которое будет использоваться при pip install
    version="0.8.0",
    author="madkiba",
    author_email="ogletix@gmail.com",
    description="Python library for Species Distribution Modeling (Python-SDM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mad-kiba/sdm-library",
    packages=find_packages(exclude=["tests", "examples"]), # Автоматически находит пакеты
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "geopandas>=0.9.0",
        "rasterio>=1.2.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Или ваша лицензия
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: GIS",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.7", # Минимальная версия Python
    #entry_points={
    #    # Если вы делаете CLI утилиту
    #    'console_scripts': [
    #        'sdm=sdm.cli.sdm_cli:main',
    #    ],
    #},
)