# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="milsf-hetnet",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="MiLSF (Minimum Load Sleep First) strategy implementation for Heterogeneous Cellular Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/milsf-hetnet",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Communications :: Telephony",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
            "streamlit>=1.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "milsf-demo=examples.basic_milsf_demo:main",
            "milsf-reproduce=examples.paper_reproduction:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/milsf-hetnet/issues",
        "Source": "https://github.com/your-username/milsf-hetnet",
        "Documentation": "https://milsf-hetnet.readthedocs.io/",
        "Paper": "https://ieeexplore.ieee.org/document/10285284",
    },
)