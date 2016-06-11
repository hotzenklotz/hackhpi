#! /bin/bash

data=$1


for class in $(ls $data); do
  for image in $(ls $data/$class); do
    convert -rotate 90 $data/$class/$image "$data/$class/${image}_90.jpg"
    convert -rotate 180 $data/$class/$image "$data/$class/${image}_180.jpg"
    convert -rotate 270 $data/$class/$image "$data/$class/${image}_270.jpg"
  done
done