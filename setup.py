from setuptools import setup, find_packages

with open('README.md') as _f:
    _README_MD = _f.read()

with open('requirements.txt') as f:
    _REQUIREMENTS = f.read().strip().splitlines()

_PACKAGES = find_packages(include=['artifici_lda.*'])

_VERSION = '1.0'

setup(
    name='artifici_lda',
    version=_VERSION,
    description='This project is for text clustering using the Latent Dirichlet Allocation (LDA) algorithm. It can be adapted to many languages provided that the Snowball stemmer, a dependency of this project, supports it.',
    long_description=_README_MD,
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: Filters",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Utilities"
    ],
    url='https://github.com/ArtificiAI/Multilingual-Latent-Dirichlet-Allocation-LDA',
    download_url='https://github.com/ArtificiAI/Multilingual-Latent-Dirichlet-Allocation-LDA/tarball/{}'.format(_VERSION),
    author='Neuraxio Inc.',
    author_email='guillaume.chevalier@neuraxio.com',
    packages=_PACKAGES,
    test_suite="testing",
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
    install_requires=_REQUIREMENTS,
    include_package_data=True,
    license='MIT',
    keywords='Multilingual Latent Dirichlet Allocation (LDA) Pipeline NLP Natural Language Processing'
)

