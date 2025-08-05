from setuptools import setup, find_packages

setup(
    name="baby_cry_classification",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "librosa",
        "tensorflow",
        "scikit-learn",
        "imblearn",
        "tqdm",
        "ipython",
        "pytest",
        "scipy"
    ],
)
