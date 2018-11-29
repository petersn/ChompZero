#!/usr/bin/python

import collections

ChompGameConfig = collections.namedtuple("ChompGameConfig", ["width", "height"])

class ChompState:
	def __init__(self, config, limits, to_move=1):
		self.limits = limits

