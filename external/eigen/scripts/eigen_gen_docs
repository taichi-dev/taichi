#!/bin/sh

# configuration
# You should call this script with USER set as you want, else some default
# will be used
USER=${USER:-'orzel'}
UPLOAD_DIR=dox-devel

#ulimit -v 1024000

# step 1 : build
rm build/doc/html -Rf
mkdir build -p
(cd build && cmake .. && make doc) || { echo "make failed"; exit 1; }

#step 2 : upload
# (the '/' at the end of path is very important, see rsync documentation)
rsync -az --no-p --delete build/doc/html/ $USER@ssh.tuxfamily.org:eigen/eigen.tuxfamily.org-web/htdocs/$UPLOAD_DIR/ || { echo "upload failed"; exit 1; }

#step 3 : fix the perm
ssh $USER@ssh.tuxfamily.org "chmod -R g+w /home/eigen/eigen.tuxfamily.org-web/htdocs/$UPLOAD_DIR" || { echo "perm failed"; exit 1; }

echo "Uploaded successfully"

