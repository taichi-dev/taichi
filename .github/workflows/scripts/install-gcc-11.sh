# This is a temporary workaround to address CI failure
# From this commit, Taichi will be built using Clang-15 and libstdc++ that shipped with g++ 11,
# meaning the resulting binary won't be compatible with a basic Ubuntu 18.04 installation due to
# the need for a more recent version of libstdc++.
# In the subsequent commit (static linking), we can eliminate this workaround.

apt install -y software-properties-common
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt update
apt install -y g++-11
