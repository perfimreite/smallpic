#!/bin/bash

main="main.py"

source ./bin/activate
echo -e "[INFO] executing \`$main\`\n"
python3 $main
deactivate

