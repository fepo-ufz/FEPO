from pathlib import Path
from setuptools import setup, find_packages

# Read README for long_description
BASE_DIR = Path(__file__).resolve().parent
README = (BASE_DIR / "README.md").read_text(encoding="utf-8")

setup(
    name="fepo",
    version="1.0.0",
    description="Sand Flies Extreme Population Prediction Model",
    long_description=README,
    long_description_content_type="text/markdown",

    author="Dr. Majid Soheili",
    author_email="majid.soheili@ufz.de",

    url="https://github.com/fepo-ufz/FEPO",
    license="LGPL-3.0-or-later",

    packages=find_packages(
        where=".",
        include=["fepo*"],
    ),

    python_requires=">=3.7",

    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "catboost",
        "scipy",
        'importlib-metadata; python_version < "3.10"',
    ],

    project_urls={
        "Bug Tracker": "",
        "Documentation": "",
        "Source Code": "https://github.com/fepo-ufz/FEPO",
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",

        "Operating System :: OS Independent",

        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",

        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    keywords="sandflies prediction model fepo machine-learning climate",
)