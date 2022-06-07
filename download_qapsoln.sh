#!/bin/zsh
sln_format="http://www.mgi.polymtl.ca/anjos/qaplib/soln.d/%s.sln"
urls=()
for f in ./qapdata/*.dat; do
    urls+=($(printf "$sln_format" "$(basename -s .dat $f)"))
done
wget --directory-prefix qapsoln $urls
