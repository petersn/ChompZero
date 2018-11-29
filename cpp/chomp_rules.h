// Chomp rules.

#ifndef CHOMP_RULES_H
#define CHOMP_RULES_H

#include <iostream>
#include <cstdint>

// For efficiency I specialize the code to one board size.
constexpr int BOARD_SIZE = 16;
constexpr int MAX_MOVE_COUNT = BOARD_SIZE * BOARD_SIZE;

typedef uint8_t ChompInt;

enum class Player {
	NOBODY = 0,
	PLAYER1 = 1,
	PLAYER2 = 2,
};

static inline Player opponent_of(Player p) {
	assert(p == Player::PLAYER1 or p == Player::PLAYER2);
	return p == Player::PLAYER1 ? Player::PLAYER2 : Player::PLAYER1;
}

struct ChompMove {
	ChompInt x, y;

	inline bool operator ==(const ChompMove& other) const {
		return x == other.x and y == other.y;
	}

	inline bool operator !=(const ChompMove& other) const {
		return x != other.x or y != other.y;
	}
};

constexpr ChompMove NO_MOVE = ChompMove{BOARD_SIZE, BOARD_SIZE};

struct ChompState {
	ChompInt limits[BOARD_SIZE];
	Player to_move;
	Player winner;

	bool is_cell_filled(const ChompMove& m) const;
	void apply_move(const ChompMove& m);
	int legal_moves(ChompMove* buf) const;
};

ChompState make_empty_state();
std::ostream& operator <<(std::ostream& os, const ChompMove& move);
std::ostream& operator <<(std::ostream& os, const ChompState& state);
void print_moves(const ChompState& state);

#endif

