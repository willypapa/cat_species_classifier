.PHONY: create_environment git

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = Cat Species Classifier
CONDA_ENVIRONMENT = cat_species_classifier
PYTHON_VERSION = 3.7


##

.PHONY: help clean clean-pyc clean-build list test test-all coverage docs release sdist

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "sdist - package"

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

lint:
	flake8 cat_species_classifier test

test:
	py.test

test-all:
	tox

#coverage:
#	coverage run --source cat_species_classifier setup.py test
#	coverage report -m
#	coverage html
#	open htmlcov/index.html

#docs:
#	sphinx-apidoc -o docs/ cat_species_classifier
#	$(MAKE) -C docs clean
#	$(MAKE) -C docs html
#	open docs/_build/html/index.html

#sdist: clean
#	pip freeze > requirements.rst
#	python setup.py sdist
#	ls -l dist


##

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
# this is done in the Dockerfile
#environment:
#	conda env create -f environment.yml

git:
	git init

##
# DOCKER
##

# https://github.com/mapbox/robosat/blob/master/Makefile
#dockerimage ?= mapbox/robosat
#dockerfile ?= Dockerfile
srcdir ?= $(shell pwd)
datadir ?= $(shell pwd)

install:
	@docker-compose up --no-start --build cat_species_classifier
	@docker start cat_species_classifier_container
#	@docker build -t $(dockerimage) -f $(dockerfile) .

i: install

update:
	@docker-compose up --no-start --build cat_species_classifier
	@docker start cat_species_classifier_container
	#@docker build -t $(dockerimage) -f $(dockerfile) . --pull --no-cache

u: update


run:
	@docker exec -it cat_species_classifier_container /bin/bash
	#@docker run -it --rm --ipc="host" --network="host" -v $(srcdir)/robosat:/usr/src/app/robosat -v $(datadir):/data --entrypoint=/bin/bash $(dockerimage)

r: run

.PHONY: install i run r update u


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) == Darwin && echo '--no-init --raw-control-chars')
