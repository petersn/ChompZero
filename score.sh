#!/bin/bash

BAYESELO=~/local/BayesElo/bayeselo
PGN=$1

$BAYESELO <<EOF
readpgn $1
elo
mm
exactdist
ratings
EOF

echo

