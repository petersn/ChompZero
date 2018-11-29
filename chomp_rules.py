#!/usr/bin/python

import collections

ChompGameConfig = collections.namedtuple("ChompGameConfig", ["width", "height"])

OTHER_PLAYER = {1: 2, 2: 1}

class ChompState:
	def __init__(self, config, limits, to_move=1, winner=None):
		self.config = config
		self.limits = limits
		self.to_move = to_move
		self.winner = winner

	@staticmethod
	def empty_board(config):
		return ChompState(
			config=config,
			limits=[config.width for y in xrange(config.height)],
			to_move=1,
		)

	def __getitem__(self, xy):
		"""__getitem__(self, (x, y)) -> bool

		Returns True iff the location is filled.
		"""
		x, y = xy
		assert 0 <= x < self.config.width
		assert 0 <= y < self.config.height
		return x >= self.limits[y]

	def __str__(self):
		return "\n".join(
			 " ".join(
			 	{False: ".", True: "#"}[self[x, y]]
				for x in xrange(self.config.width)
			 )
			 for y in reversed(xrange(self.config.height))
		)

	def copy(self):
		return ChompState(
			self.config,
			self.limits[:],
			self.to_move,
			self.winner,
		)

	def apply_move(self, xy):
		assert not self[xy]
		x, y = xy
		for hit_y in xrange(y, self.config.height):
			self.limits[hit_y] = min(self.limits[hit_y], x)
		# Check if see if the player ate the poisoned chocolate.
		if xy == (0, 0):
			self.winner = OTHER_PLAYER[self.to_move]
		# Flip whose turn it is.
		self.to_move = OTHER_PLAYER[self.to_move]

	def legal_moves(self):
		for y in xrange(self.config.height):
			for x in xrange(self.limits[y]):
				yield x, y

if __name__ == "__main__":
	import random
	while True:
		print "Launching game."
		config = ChompGameConfig(4, 4)
		board = ChompState.empty_board(config)
		while list(board.legal_moves()):
			print board
			print board.winner, board.to_move
			move = random.choice(list(board.legal_moves()))
			board.apply_move(move)
			raw_input()

