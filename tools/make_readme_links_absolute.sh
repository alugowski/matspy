#!/bin/bash

cat $(dirname "$0")/../README.md |
sed -e 's,doc/,https://github.com/alugowski/matspy/blob/main/doc/,g' |
sed -e 's,(demo-,(https://nbviewer.org/github/alugowski/matspy/blob/main/demo-,g'