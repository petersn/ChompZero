#!/usr/bin/python

import sys, string
import chomp_rules
import engine

def uai_encode_move(move):
	x, y = move
	return "%i,%i" % (x, y)

def uai_decode_move(s):
	x, y = s.split(",")
	return int(x), int(y)

def main(args):
	config = chomp_rules.ChompGameConfig(engine.model.BOARD_SIZE, engine.model.BOARD_SIZE)
	board = chomp_rules.ChompState.empty_board(config)
	eng = engine.MCTSEngine()
	if args.visits != None:
#		eng.VISITS = args.visits
		eng.MAX_STEPS = args.visits

	while True:
		line = raw_input()
		if line == "quit":
			exit()
		elif line == "uai":
			print "id name ChompZero"
			print "id author Peter Schmidt-Nielsen"
			print "uaiok"
		elif line == "uainewgame":
			board = chomp_rules.ChompState.empty_board(config)
			eng = engine.MCTSEngine()
		elif line.startswith("moves "):
			for move in line[6:].split():
				move = uai_decode_move(move)
				board.move(move)
			eng.set_state(board.copy())
		elif line.startswith("go movetime "):
			ms = int(line[12:])
			if args.visits == None:
				move = eng.genmove(ms * 1e-3, use_weighted_exponent=5.0)
			else:
				# This is safe, because of the visit limit we set above.
				move = eng.genmove(1000000.0, use_weighted_exponent=5.0)
			print "bestmove %s" % (uai_encode_move(move),)
		elif line == "showboard":
			print board
			print "boardok"
		sys.stdout.flush()

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--network-path", metavar="NETWORK", type=str, help="Name of the model to load.")
	parser.add_argument("--visits", metavar="VISITS", default=None, type=int, help="Number of visits during MCTS.")
	args = parser.parse_args()
	print >>sys.stderr, args

	engine.setup_evaluator(use_rpc=False)
	engine.initialize_model(args.network_path)
	main(args)

