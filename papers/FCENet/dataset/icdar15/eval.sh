#!/bin/bash

cd dataset/icdar15/
rm submit/*
cp ../../output/Icdar2015/*.txt submit
cd submit
zip -q -r submit.zip *
mv submit.zip ../
cd ..
python Evaluation_Protocol/script.py -g=gt.zip -s=submit.zip
cd ../..