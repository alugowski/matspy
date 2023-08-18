#!/bin/bash

cat $(dirname "$0")/../README.md |
sed -e 's,doc/,https://github.com/alugowski/matspy/blob/main/doc/,g' |
sed -e 's,(demo-,(https://github.com/alugowski/matspy/blob/main/demo-,g'