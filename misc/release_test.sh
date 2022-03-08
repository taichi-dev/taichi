#!/usr/bin/env bash

# Taichi release test suite

# This script is created mainly for eyeball-testing
# that if all of the official examples are still working
# with the latest version of Taichi.

# The test cases are created from external repositories
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
        python "${match}"
        taichi::utils::line
        taichi::utils::pause
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
        python "${match}"
        taichi::utils::line
        taichi::utils::pause
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
        python "${match}"
        taichi::utils::line
        taichi::utils::pause
    done

    # run special tests
    python3 render_particles.py -i ./path/to/particles \
                                -b 0 -e 400 -s 1 \
                                -o ./output \
                                --gpu-memory 20 \
                                -M 460 \
                                --shutter-time 0.0 \
                                -r 128

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
    python src/main.py

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

    # stannum tests 
    taichi::test::sandyfluid "${WORKDIR}"
}

taichi::test::main
