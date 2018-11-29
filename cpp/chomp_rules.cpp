// Chomp rules.

#include <cassert>
#include "chomp_rules.h"

bool ChompState::is_cell_filled(const ChompMove& m) const {
	assert(m.x < BOARD_SIZE);
	assert(m.y < BOARD_SIZE);
	return m.x > limits[m.y];
}

void ChompState::apply_move(const ChompMove& m) {
	assert(not is_cell_filled(m));
	for (int hit_y = m.y; hit_y < BOARD_SIZE; hit_y++)
		limits[hit_y] = std::min(limits[hit_y], m.x);
	if (m.x == 0 and m.y == 0)
		winner = opponent_of(to_move);
	to_move = opponent_of(to_move);
}

int ChompState::legal_moves(ChompMove* buf) const {
	int move_count = 0;
	for (ChompInt y = 0; y < BOARD_SIZE; y++) {
		for (ChompInt x = 0; x < limits[y]; x++) {
			*buf++ = ChompMove{x, y};
			move_count++;
		}
	}
	return move_count;
}

ChompState make_empty_state() {
	ChompState result;
	std::fill(result.limits, result.limits + BOARD_SIZE, BOARD_SIZE);
	result.to_move = Player::PLAYER1;
	result.winner = Player::NOBODY;
	return result;
}

std::ostream& operator <<(std::ostream& os, const ChompMove& move) {
	return os << "{" << static_cast<int>(move.x) << ", " << static_cast<int>(move.y) << "}";
}

std::ostream& operator <<(std::ostream& os, const ChompState& state) {
	for (ChompInt y_inv = 0; y_inv < BOARD_SIZE; y_inv++) {
		ChompInt y = (BOARD_SIZE - 1) - y_inv;
		for (ChompInt x = 0; x < BOARD_SIZE; x++) {
			os << (state.is_cell_filled({x, y}) ? "#" : ".");
			if (x < BOARD_SIZE - 1)
				os << " ";
		}
		os << "\n";
	}
	return os;
}

void print_moves(const ChompState& state) {
	ChompMove moves[MAX_MOVE_COUNT];
	int move_count = state.legal_moves(moves);
	for (int i = 0; i < move_count; i++) {
		std::cout << moves[i];
		if (i < move_count - 1)
			std::cout << " ";
		std::cout << std::endl;
	}
}

