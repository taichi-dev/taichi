#!/usr/bin/env bash

# Taichi release test suite

# Usage: `bash release_test.sh`

# This script is created mainly for eyeball-testing
# that if all of the official examples are still working
# with the latest version of Taichi.

# Some of the test cases are fetched from external repositories
# please reach out to us if you are the owner of those
# repos and don't like us to do it.

# You can add more tests into this script and plug-n-play
# existing tests in the `taichi::test::main` function as
# you need.

function taichi::utils::set_debug {
    if [ ${DEBUG} == "true" ]; then
        set -x
    fi
    set -euo pipefail
}

function taichi::utils::logger {
    # default: gray
    if [ "$1" == "info" ]; then
        printf '\e[1;90m%-6s\e[m\n' "$(date +"[%m-%d %H:%M:%S]") $2"
    # error: red
    elif [ "$1" == "error" ]; then
        printf '\e[1;91m%-6s\e[m\n' "$(date +"[%m-%d %H:%M:%S]") $2"
    # success: green
    elif [ "$1" == "success" ]; then
        printf '\e[1;92m%-6s\e[m\n' "$(date +"[%m-%d %H:%M:%S]") $2"
    # warning: yellow
    elif [ "$1" == "warning" ]; then
        printf '\e[1;93m%-6s\e[m\n' "$(date +"[%m-%d %H:%M:%S]") $2"
    # debug: gray
    elif [ "$1" == "debug" ]; then
        if [ "${DEBUG}" == "true" ]; then
            printf '\e[1;90m%-6s\e[m\n' "$(date +"[%m-%d %H:%M:%S]") $2"
        fi
    else
        printf "$1"
    fi
}

function taichi::utils::logger::info {
    taichi::utils::logger "info" "$1"
}

function taichi::utils::logger::error {
    taichi::utils::logger "error" "$1"
}

function taichi::utils::logger::success {
    taichi::utils::logger "success" "$1"
}

function taichi::utils::logger::warning {
    taichi::utils::logger "warning" "$1"
}

function taichi::utils::logger::debug {
    taichi::utils::logger "debug" "$1"
}

function taichi::utils::line {
    printf '%.0s-' {1..20}; echo
}

function taichi::utils::git_clone {
    local GIT_ORG=$1
    local GIT_REPO=$2
    git clone "git@github.com:${GIT_ORG}/${GIT_REPO}.git"
}

function taichi::utils::pause {
    read -p "Press enter to continue"
}

function taichi::utils::pkill {
    sleep 5
    pkill -f "$1"
}

function taichi::test::ggui {
    local WORKDIR=${1}
    local PATTERN="*_ggui.py"
    local ORG="taichi-dev"
    local REPO="taichi"

    # divider
    taichi::utils::line
    taichi::utils::logger::info "Running GGUI examples"

    # clone the repo
    taichi::utils::git_clone "${ORG}" "${REPO}"
    cd "${REPO}/python/taichi/examples/ggui_examples"

    # run tests
    for match in $(find ./ -name "${PATTERN}"); do
        python "${match}" &
        taichi::utils::pkill "${match}"
        taichi::utils::line
        # taichi::utils::pause
    done

    # go back to workdir
    cd "${WORKDIR}"
}

function taichi::test::difftaichi {
    local WORKDIR=${1}
    local PATTERN="*.py"
    local ORG="taichi-dev"
    local REPO="difftaichi"

    # divider
    taichi::utils::line
    taichi::utils::logger::info "Running DiffTaichi examples"

    # clone the repo
    taichi::utils::git_clone "${ORG}" "${REPO}"
    cd "${REPO}/examples"

    # run tests
    for match in $(find ./ -name "${PATTERN}"); do
        python "${match}" &
        taichi::utils::pkill "${match}"
        taichi::utils::line
        # taichi::utils::pause
    done

    # go back to workdir
    cd "${WORKDIR}"
}

function taichi::test::taichi_elements {
    local WORKDIR=${1}
    local PATTERN="demo_*.py"
    local ORG="taichi-dev"
    local REPO="taichi_elements"

    # divider
    taichi::utils::line
    taichi::utils::logger::info "Running Taichi Elements examples"

    # clone the repo
    taichi::utils::git_clone "${ORG}" "${REPO}"
    cd "${REPO}"

    # install dependencies
    python "download_ply.py"


    # run tests
    cd "${REPO}/demo"
    for match in $(find ./ -name "${PATTERN}"); do
        python "${match}" &
        taichi::utils::pkill "${match}"
        taichi::utils::line
        # taichi::utils::pause
    done

    # run special tests
    # FIXME: this does not work properly yet
    # taichi::utils::logger::success $(ls)
    # read -p "Please input the directory containing the generated particles, e.g. sim_2022-01-01_20-55-48" particles_dir
    # python render_particles.py -i ./"${particles_dir}" \
    #                            -b 0 -e 400 -s 1 \
    #                            -o ./output \
    #                            --gpu-memory 20 \
    #                            -M 460 \
    #                            --shutter-time 0.0 \
    #                            -r 128

    # go back to workdir
    cd "${WORKDIR}"
}

function taichi::test::stannum {
    local WORKDIR=${1}
    local ORG="ifsheldon"
    local REPO="stannum"

    # divider
    taichi::utils::line
    taichi::utils::logger::info "Running Stannum examples"

    # clone the repo
    taichi::utils::git_clone "${ORG}" "${REPO}"
    cd "${REPO}"

    # run tests
    pytest -v -s ./
    taichi::utils::line

    # go back to workdir
    cd "${WORKDIR}"
}

function taichi::test::sandyfluid {
    local WORKDIR=${1}
    local ORG="ethz-pbs21"
    local REPO="SandyFluid"

    # divider
    taichi::utils::line
    taichi::utils::logger::info "Running SandyFluid examples"

    # clone the repo
    taichi::utils::git_clone "${ORG}" "${REPO}"
    cd "${REPO}"

    # install dependencies
    # remove the line contains pinned Taichi version for testing purposes
    grep -v "taichi" requirements.txt > tmpfile && mv tmpfile requirements.txt
    pip install -r requirements.txt

    # run tests
    python src/main.py &
    taichi::utils::pkill "src/main.py"
    taichi::utils::line
    # go back to workdir
    cd "${WORKDIR}"
}

function taichi::test::voxel_editor {
    local WORKDIR=${1}
    local ORG="taichi-dev"
    local REPO="voxel_editor"

    # divider
    taichi::utils::line
    taichi::utils::logger::info "Running Voxel Editor examples"

    # clone the repo
    taichi::utils::git_clone "${ORG}" "${REPO}"
    cd "${REPO}"

    # run tests
    python voxel_editor.py &
    taichi::utils::pkill "voxel_editor.py"
    taichi::utils::line

    # go back to workdir
    cd "${WORKDIR}"
}

function taichi::test::generate_videos {
    local WORKDIR=${1}
    local PATTERN="test_*.py"
    local ORG="taichi-dev"
    local REPO="taichi"

    # divider
    taichi::utils::line
    taichi::utils::logger::info "Generating examples videos"

    # clone the repo
    taichi::utils::git_clone "${ORG}" "${REPO}"
    # mkdir "${REPO}/misc/output_videos"

    # run tests
    cd "${REPO}/tests/python/examples"
    for directory in $(find ./ -mindepth 1 -maxdepth 1 -name "*" ! -name "__*" -type d); do
        cd "${directory}"
        for match in $(find ./ -maxdepth 1 -name "${PATTERN}" -type f); do
            pytest -v "${match}"
            taichi::utils::line
            # taichi::utils::pause
        done
        cd ..
    done

    # go back to workdir
    cd "${WORKDIR}"
}

function taichi::test::main {
    # set debugging flag
    DEBUG="false"

    # create a temporary directory for testing
    WORKDIR="$(mktemp -d)"
    taichi::utils::logger::info "Running all tests within ${WORKDIR}"

    # make sure to clean up the temp dir on exit
    trap '{ rm -rf -- "$WORKDIR"; }' EXIT

    # walk into the working dir
    cd "${WORKDIR}"

    # ggui examples
    taichi::test::ggui "${WORKDIR}"

    # difftaichi examples
    taichi::test::difftaichi "${WORKDIR}"

    # taichi_elements examples
    taichi::test::taichi_elements "${WORKDIR}"

    # stannum tests
    taichi::test::stannum "${WORKDIR}"

    # sandyfluid tests
    taichi::test::sandyfluid "${WORKDIR}"

    # voxel editor tests
    taichi::test::voxel_editor "${WORKDIR}"

    # generating example videos
    taichi::test::generate_videos "${WORKDIR}"
}

taichi::test::main
