# Every variable in subdir must be prefixed with subdir (emulating a namespace)
PYTHON_ENV = $(OUT)/python_env

# Each module has a top level target as the entrypoint which must match the subdir name
python_env: $(PYTHON_ENV)/.installed

python_env/dev: $(PYTHON_ENV)/.installed-dev

python_env/clean:
	rm -rf $(PYTHON_ENV)

# .PRECIOUS: $(PYTHON_ENV)/.installed $(PYTHON_ENV)/%
$(PYTHON_ENV)/.installed:
	python3.8 -m venv $(PYTHON_ENV)
	bash -c "source $(PYTHON_ENV)/bin/activate && python -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu"
	echo "Installing python env build backend requirements..."
	bash -c "source $(PYTHON_ENV)/bin/activate && python -m pip install setuptools wheel"
	touch $@

$(PYTHON_ENV)/%: $(PYTHON_ENV)/.installed
	bash -c "source $(PYTHON_ENV)/bin/activate"

$(PYTHON_ENV)/.installed-dev: tt_eager/tt_lib/dev_install python_env tt_metal/python_env/requirements-dev.txt
	echo "Installing dev environment packages..."
	bash -c "source $(PYTHON_ENV)/bin/activate && python -m pip install -r tt_metal/python_env/requirements-dev.txt"
	touch $@
