from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="miceforest",
    version="2.0.6",
    author="Samuel Wilson",
    license="MIT",
    author_email="samwilson303@gmail.com",
    test_suite="tests",
    description="Imputes missing data with MICE + random forests",
    keywords=['MICE','Imputation','Missing Values','Missing','Random Forest'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['scikit-learn',
                      'numpy',
                      'pandas',
                      'seaborn >= 0.11.0',
                      'matplotlib >= 3.3.0'
                      ],
    url="https://github.com/AnotherSamWilson/miceforest",
    packages=find_packages(exclude=["tests.*", "tests"]),
    classifiers=[
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
