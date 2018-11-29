#!/bin/bash

BAYESELO=~/local/BayesElo/bayeselo
PGN=run2/tournament/games.pgn

sed "s/python uai_interface.py //g" $PGN > /tmp/tmp.pgn

$BAYESELO >/tmp/scores <<EOF
readpgn /tmp/tmp.pgn
elo
mm
exactdist
ratings
EOF

cat /tmp/scores
echo

python make_plot.py /tmp/scores

eog /tmp/output.png

