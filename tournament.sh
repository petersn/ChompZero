#!/bin/bash

ROOT=run1

time ./uai_ringmaster.py \
	--pgn-out $ROOT/tournament/games.pgn \
	--max-plies 500 \
	--model-pattern "run1/models/model-%03i"

