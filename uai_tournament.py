#!/usr/bin/python

import os, re, collections, random
#import os, sys, random, string, subprocess, atexit, time, datetime, itertools
import chess.pgn
import uai_ringmaster
#import chomp_rules
#import model

def enumerate_engines(args):
	engines = []
	model_number = 1
	while True:
		test_path = args.model_pattern % (model_number,)
		if os.path.exists(test_path):
			engines.append((
				"python", "uai_interface.py",
					"--visits", "1000",
					"--network-path", test_path,
			))
			model_number += 1
		else:
			return engines

def get_model_number(s):
	return int(re.search("model-([0-9]+)", s).groups()[0])

def collect_games_so_far(args):
	match_ups = collections.Counter()
	players = set()
	with open(args.pgn_out) as f:
		while True:
			game = chess.pgn.read_game(f)
			if game is None:
				break
			p1 = get_model_number(game.headers["White"])
			p2 = get_model_number(game.headers["Black"])
			players.add(p1)
			players.add(p2)
			assert p1 != p2
			match_ups[p1, p2] += 1
	for p1 in players:
		for p2 in players:
			if p1 == p2:
				continue
			assert match_ups[p1, p2] == match_ups[p2, p1], \
				"Unequal number of games %i <-> %i" % (p1, p2)
	return match_ups

if __name__ == "__main__":
	import argparse, shlex
	parser = argparse.ArgumentParser()
	parser.add_argument("--show-games", action="store_true", help="Show the games while they're being generated.")
	parser.add_argument("--opening", metavar="MOVES", type=str, default=None, help="Comma separated sequence of moves for the opening.")
	parser.add_argument("--max-plies", metavar="N", type=int, default=None, help="Maximum number of plies in a game before it's aborted and rejected.")
	parser.add_argument("--pgn-out", metavar="PATH", type=str, default=None, help="PGN file path to accumulate games into. Writes in append mode.")
	parser.add_argument("--tc", metavar="SEC", type=float, default=1.0, help="Seconds per move for all engines.")
	parser.add_argument("--model-pattern", metavar="PATTERN", type=str, help="Pattern containing a %i that should be used to find the various models.")
	args = parser.parse_args()
	print "Options:", args

	games_written = 0

	while True:
		engines = enumerate_engines(args)
		print "Engines:"
		for i, engine in enumerate(engines):
			print "%4i: %s" % (i + 1, engine)

		model_number_to_engine = {}
		for engine in engines:
			model_number_to_engine[get_model_number(" ".join(engine))] = engine

		matchup_counts = collect_games_so_far(args)
		# Add in zeros for all possible match-ups.
		for p1 in model_number_to_engine:
			for p2 in model_number_to_engine:
				matchup_counts[p1, p2] += 0
		# Filter down to match-ups that are only in one order, within a closed radius of 10.
		matchup_counts = {
			(p1, p2): v for (p1, p2), v in matchup_counts.iteritems()
			if p1 < p2 and abs(p1 - p2) <= 10
		}

		# Pick an arbitrary pairing that has as few games as are known.
		min_games = min(matchup_counts.itervalues())
		frontier = [matchup for matchup, count in matchup_counts.iteritems() if count == min_games]
		print "Min games:", min_games, "Frontier:", frontier

		p1, p2 = random.choice(frontier)
		opening = uai_ringmaster.get_opening(args)

		eng1 = model_number_to_engine[p1]
		eng2 = model_number_to_engine[p2]

		game1 = uai_ringmaster.play_one_game(args, eng1, eng2, opening_moves=opening)
		game2 = uai_ringmaster.play_one_game(args, eng2, eng1, opening_moves=opening)

		if args.pgn_out:
			print "Writing games."
			write_game_to_pgn(args, args.pgn_out, game1, round_index=games_written + 1)
			write_game_to_pgn(args, args.pgn_out, game2, round_index=games_written + 2)
			games_written += 2

