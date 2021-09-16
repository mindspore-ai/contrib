# CS-F-LTR

This is the MindSpore implementation of CS-F-LTR in the following paper.

ICDE 2021: An Efﬁcient Approach for Cross-Silo Federated Learning to Rank

The paper is located in ./ICDE21-wang.pdf

# Contents

- [CS-F-LTR Description](#cs-f-ltr-description)
- [Dataset](#dataset)
- [Directory](#directory)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Mapper](#mapper)
    - [Builder](#builder)
    - [Server](#server)
    - [Operations](#operations)

# [CS-F-LTR-description](#Contents)

CS-F-LTR is a noval framework named Cross-Silo Federated Learning-to-Rank.

# [Dataset](#Contents)

Dataset used: [MS MARCO Ranking dataset](<https://www.msmarco.org/dataset.aspx>)

- Dataset partition：
    - 4 parties
    - 200 queries and 36,400 documents per party
    - 1000 terms per document
    - Train
        - 28,000 instances per party
    - Test
        - 32,000 instances per party

# [Directory](#Contents)

- raw docs are split into txt in ./data/docs
- raw queries are split into txt in ./data/queries
- all term have been numbered and the dictionary is in ./data

# [Quick Start](#Contents)

```bash
  bash scripts/run.sh
```

# [Script Description](#Contents)

## [Mapper](#Contents)

- transfer all raw docs queries and top100 into mapper
- the docs and queries and relevance docs and score are all in ./data/mapper{TOP_NUM}

## [Builder](#Contents)

based on the data transferred by mapper, build {FED_NUM} federations

## [Server](#Contents)

this script helps to exchange most relevance features and upgrade each federation
then each federation will gen features

## [Operations](#Contents)

```bash
python3 dictionary.py (optional)
python3 mapper.py
python3 builder.py -b 1
python3 builder.py -b 2 -f 0
python3 builder.py -b 2 -f 1
python3 builder.py -b 2 -f 2
python3 builder.py -b 2 -f 3
python3 server.py -f 0
python3 server.py -f 1
python3 server.py -f 2
python3 server.py -f 3
```
