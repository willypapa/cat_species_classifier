Cat Species Classifier
==============================

Classify cat species

Project Organization
------------
    .
    ├── bin/                                  <- Executable scripts directory│   ├── connect_to_aws                    <- Connect to AWS EC2 instance
    │   ├── connect_to_notebook               <- Open a connection to jupyter notebook port on EC2 instance
    │   └── update_ssh_config                  <- Update your ssh config file with current instance IP

    │
    ├── config.yml                             <- configuration file for global project variables
    │
    ├── cat_species_classifier <- The python package, **install with 'pip install -e .'**
    │   ├── config.py                          <- Makes config.yml variables accessible in cat_species_classifier.config namespace
    │   └── models/                           <- store ML models
    │
    ├── data                                  <- Data dump
    │   ├── intermediate                      <- Intermediate data e.g. serialised arrays
    │   ├── processed                         <- Data ready-for-training/inference
    │   ├── raw                               <- Raw data - should be immutable
    │   └── volumes                           <- Mirrored drives
    │
    ├── docker-compose.yml          <- Start-up configuration for docker container
    │
    ├── Dockerfile                   <- Instructions for building docker image
    │
    ├── envs                        <- conda virtual environment definition files
    │   ├── cat_species_classifier_cpu_env.yml
    │   └── cat_species_classifier_gpu_env.yml
    │
    ├── figures                      <- Figures saved by scripts or notebooks.
    │
    ├── LICENSE
    │
    ├── Makefile                     <- Makefile with commands like `make environment`
    │
    ├── notebooks/                  <- Jupyter notebooks
    │
    ├── output/                     <- Manipulated data, logs, etc.
    │
    ├── README.md                   <- The top-level README for developers using this project.
    │
    ├── setup.py
    │
    ├── tests                       <- Unit tests.
    │
    └── tox.ini                     <- tox file with settings for running tox; see tox.testrun.org


--------

Project based on the [cookiecutter data science project template][https://drivendata.github.io/cookiecutter-data-science/]

Set up
------------

Maybe make this a git repo:

```bash
$ git init
```

Install the docker container and embedded conda environment:

```bash
$ make install
```

Enter the docker container (ensure script executable e.g. `chmod 755 ./bin/enter_container`):

```bash
./bin/enter_container
```

Activate the conda environment:

```bash
$ conda activate cat_species_classifier
```

If you haven't done this in the container already, install the local python module:

```bash
$ pip install -e .
```
