poetry run yapf -ir engine/
poetry run yapf -ir tests/
poetry run mypy --ignore-missing-imports .
poetry run pylint engine
poetry run pylint tests
