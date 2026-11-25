# list all recipes
default:
  @just --list --unsorted

# ------------------------------------------------------------------------------
# manage development environment
# ------------------------------------------------------------------------------
# setup local development environment with uv
sync:
  uv sync

# install pre-commit hooks
install-pre-commit: sync
  uv run pre-commit install

# ------------------------------------------------------------------------------
# linting and formatting
# ------------------------------------------------------------------------------
# lint code with ruff
lint: sync
  uv run ruff check

# fix linting issues with ruff
fix: sync
  uv run ruff check --fix

# format code with ruff
format: sync
  uv run ruff format

# check (format and lint) code with ruff
check: sync
  uv run ruff format
  uv run ruff check --fix

# ------------------------------------------------------------------------------
# running stuff
# ------------------------------------------------------------------------------
# run python src/ entrypoint
run target *args: sync
  PYTHONPATH=src uv run {{target}} {{args}}

# ------------------------------------------------------------------------------
# docker
# ------------------------------------------------------------------------------
docker_args := '--gpus all --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 '
docker_mounts := ('-v ' + justfile_directory() + '/src:/app/src '
                + '-v ' + justfile_directory() + '/experiments:/app/experiments '
                + '-v /datasets:/datasets ')
docker_image_name := 'j3pa:latest'
docker_run := 'docker run ' + docker_args + docker_mounts + docker_image_name

# build docker image
build-docker:
  docker build -f Dockerfile -t {{docker_image_name}} . --progress plain

# run bash in docker container
run-docker-bash:
  {{docker_run}} bash

# run python src/ entrypoint in docker container
run-docker target *args:
  {{docker_run}} python src/{{target}} {{args}}