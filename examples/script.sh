#!/bin/bash

# 设置 Python 文件所在的目录
DIRECTORY="."

# 遍历目录下的所有 Python 文件
for file in $DIRECTORY/*.py; do
    echo "Running $file..."
    python "$file"
    if [ $? -eq 0 ]; then
        echo "$file - Success"
    else
        echo "$file - Failed"
    fi
done

