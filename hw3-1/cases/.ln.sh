#!/bin/sh
DIR=/share/data/.testcases/hw3-1

for dir in $DIR/*.1*; do
        echo $dir
        echo $(basename "${dir}")
        ln -s $dir $(basename "${dir}")
done
for dir in $DIR/*.2*; do
        echo $dir
        echo $(basename "${dir}")
        ln -s $dir $(basename "${dir}")
done
