// Game generation RPC client in C++.

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <list>
#include <queue>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cmath>
#include <cassert>

#include <json.hpp>
#include "chomp_rules.h"

//#define ONE_RANDOM_MOVE

using json = nlohmann::json;
using std::shared_ptr;
using std::cout;
using std::endl;

constexpr double exploration_parameter = 3.0;
constexpr double dirichlet_alpha = 0.15;
constexpr double dirichlet_weight = 0.25;
constexpr int maximum_game_plies = 400;

constexpr int FEATURES_SIZE = BOARD_SIZE * BOARD_SIZE * 2;
constexpr int POSTERIOR_SIZE = BOARD_SIZE * BOARD_SIZE;

// Ugh, I hate the 32-bit seeding here... :(
std::random_device rd;
std::default_random_engine generator(rd());

// We raise this exception in worker threads when they're done.
struct StopWorking : public std::exception {};

// Configuration.
int global_visits;

// Extend ChompMove with a hash.
namespace std {
	template<> struct hash<ChompMove> {
		size_t operator()(const ChompMove& m) const {
			return static_cast<size_t>(m.x) + static_cast<size_t>(m.y) * BOARD_SIZE;
		}
	};
}

template<int s1, int s2, int s3>
int stride_index(int x, int y, int z) {
	assert(0 <= x and x < s1);
	assert(0 <= y and y < s2);
	assert(0 <= z and z < s3);
	return
		s2 * s3 * x +
		     s3 * y +
			      z;
}

struct pair_hash {
public:
	template <typename T, typename U>
	size_t operator()(const std::pair<T, U>& x) const {
		return std::hash<T>()(x.first) + std::hash<U>()(x.second) * 7;
	}
};

std::vector<int> serialize_board_for_json(const ChompState& board) {
	std::vector<int> result;
	for (ChompInt y = 0; y < BOARD_SIZE; y++)
		result.push_back(board.limits[y]);
	return result;
}

int get_board_result(const ChompState& board) {
	return static_cast<int>(board.winner);
}

std::pair<const float*, double> request_evaluation(int thread_id, const float* feature_string);

struct Evaluations {
	bool game_over;
	double value;
	std::unordered_map<ChompMove, double> posterior;

	void populate(int thread_id, const ChompState& board, bool use_dirichlet_noise) {
		// Completely reset the evaluation.
		posterior.clear();
		game_over = false;

		int result = get_board_result(board);

		if (result != 0) {
			game_over = true;
			// Score from the perspective of the current player.
			assert(result == 1 or result == 2);
			value = result == 1 ? 1.0 : -1.0;
			// Flip the value to be from the perspective of the current player.
			// TODO: XXX: Validate that I got this the right way around.
			if (board.to_move == Player::PLAYER2)
				value *= -1;
			return;
		}

		// Build a features map, initialized to all zeros.
		float feature_buffer[FEATURES_SIZE] = {};
		for (int y = 0; y < BOARD_SIZE; y++) {
			for (int x = 0; x < BOARD_SIZE; x++) {
				// Fill in layer 0 with all ones.
				feature_buffer[stride_index<BOARD_SIZE, BOARD_SIZE, 2>(x, y, 0)] = 1.0;
				if (board.is_cell_filled({x, y}))
					feature_buffer[stride_index<BOARD_SIZE, BOARD_SIZE, 2>(x, y, 1)] = 1.0;
			}
		}

		auto request_result = request_evaluation(thread_id, feature_buffer);
		const float* posterior_array = request_result.first;
		value = request_result.second;

		// Softmax the posterior array.
		double softmaxed[BOARD_SIZE * BOARD_SIZE];
		for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++)
			softmaxed[i] = exp(posterior_array[i]);
		double total = 0.0;
		for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++)
			total += softmaxed[i];
		if (total != 0.0) {
			for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++)
				softmaxed[i] /= total;
		}

		ChompMove moves_buffer[MAX_MOVE_COUNT];
		int num_moves = board.legal_moves(moves_buffer);

		// Evaluate movegen.
		double total_probability = 0.0;
		for (int i = 0; i < num_moves; i++) {
			ChompMove& move = moves_buffer[i];
			double probability = softmaxed[stride_index<BOARD_SIZE, BOARD_SIZE, 1>(move.x, move.y, 0)];
			posterior.insert({move, probability});
			total_probability += probability;
		}
		// Normalize the posterior.
		if (total_probability != 0.0) {
//			cout << "Total prob: " << total_probability << " Move count:" << num_moves << endl;
			for (auto& p : posterior)
				p.second /= total_probability;
		}

		assert(num_moves > 0);
		assert(posterior.size() > 0);

		// Add Dirichlet noise.
		if (use_dirichlet_noise) {
			std::gamma_distribution<double> distribution(dirichlet_alpha, 1.0);
			std::vector<double> dirichlet_distribution;
			for (auto& p : posterior)
				dirichlet_distribution.push_back(distribution(generator));
			// Normalize the Dirichlet distribution.
			double total = 0.0;
			for (double x : dirichlet_distribution)
				total += x;
			for (double& x : dirichlet_distribution)
				x /= total;
			// Perform a weighted sum of this Dirichlet distribution and our previous posterior.
			int index = 0;
			for (auto& p : posterior)
				p.second = dirichlet_weight * dirichlet_distribution[index++] + (1.0 - dirichlet_weight) * p.second;
			// Assert approximate normalization.
			double test_total = 0.0;
			for (auto& p : posterior)
				test_total += p.second;
//			assert(0.99 < test_total and test_total < 1.01);
		}
	}
};

struct MCTSEdge;
struct MCTSNode;

struct MCTSEdge {
	ChompMove edge_move;
	MCTSNode* parent_node;
	shared_ptr<MCTSNode> child_node;
	double edge_visits = 0;
	double edge_total_score = 0;

	MCTSEdge(ChompMove edge_move, MCTSNode* parent_node, shared_ptr<MCTSNode> child_node)
		: edge_move(edge_move), parent_node(parent_node), child_node(child_node) {}

	double get_edge_score() const {
		if (edge_visits == 0)
			return 0;
		return edge_total_score / edge_visits;
	}

	void adjust_score(double new_score) {
		edge_visits += 1;
		edge_total_score += new_score;
	}
};

struct MCTSNode {
	ChompState board;
	bool evals_populated = false;
	Evaluations evals;
	shared_ptr<MCTSNode> parent;
	int all_edge_visits = 0;
	std::unordered_map<ChompMove, MCTSEdge> outgoing_edges;

	MCTSNode(const ChompState& board) : board(board) {}

	double total_action_score(const ChompMove& m) {
		assert(evals_populated);
		const auto it = outgoing_edges.find(m);
		double u_score, Q_score;
		if (it == outgoing_edges.end()) {
			u_score = sqrt(1 + all_edge_visits);
			Q_score = 0;
		} else {
			const MCTSEdge& edge = (*it).second;
			u_score = sqrt(1 + all_edge_visits) / (1 + edge.edge_visits);
			Q_score = edge.get_edge_score();
		}
		u_score *= exploration_parameter * evals.posterior.at(m);
		return u_score + Q_score;
	}

	void populate_evals(int thread_id, bool use_dirichlet_noise=false) {
		if (evals_populated)
			return;
		evals.populate(thread_id, board, use_dirichlet_noise);
		evals_populated = true;
	}

	ChompMove select_action() {
		assert(evals_populated);
		// If we have no legal moves then return a null move.
		if (evals.posterior.size() == 0)
			return NO_MOVE;
		// If the game is over then return a null move.
		if (evals.game_over)
			return NO_MOVE;
		// Find the best move according to total_action_score.
		ChompMove best_move = NO_MOVE;
		double best_score = -1;
		bool flag = false;
		for (const auto p : evals.posterior) {
			double score = total_action_score(p.first);
			if (p.first == NO_MOVE) {
				std::cerr << "Bad posterior had NO_MOVE in moves list." << endl;
				std::cerr << "Posterior length: " << evals.posterior.size() << endl;
				std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
			}
			assert(not std::isnan(score));
			assert(score >= 0.0);
			if (score >= best_score) {
				best_move = p.first;
				best_score = score;
				flag = true;
			}
		}
		if (best_move == NO_MOVE) {
			std::cerr << "                  WARNING Best move is no move!" << endl;
			std::cerr << "       Posterior size: " << evals.posterior.size() << endl;
			std::cerr << "     Flag status: " << flag << " END FLAG" << endl;
		}
		return best_move;
	}
};

struct MCTS {
	int thread_id;
	ChompState root_board;
	bool use_dirichlet_noise;
	shared_ptr<MCTSNode> root_node;

	MCTS(int thread_id, const ChompState& root_board, bool use_dirichlet_noise)
		: thread_id(thread_id), root_board(root_board), use_dirichlet_noise(use_dirichlet_noise)
	{
		init_from_scratch(root_board);
	}

	void init_from_scratch(const ChompState& root_board) {
		root_node = std::make_shared<MCTSNode>(root_board);
		root_node->populate_evals(thread_id, use_dirichlet_noise);
	}

	std::tuple<shared_ptr<MCTSNode>, ChompMove, std::vector<MCTSEdge*>> select_principal_variation(bool best=false) {
		shared_ptr<MCTSNode> node = root_node;
		std::vector<MCTSEdge*> edges_on_path;
		ChompMove move = NO_MOVE;
		while (true) {
			if (best) {
				// Pick the edge that has the highest visit count.
				auto it = std::max_element(
					node->outgoing_edges.begin(),
					node->outgoing_edges.end(),
					[](
						const std::pair<ChompMove, MCTSEdge>& a,
						const std::pair<ChompMove, MCTSEdge>& b
					) {
						return a.second.edge_visits < b.second.edge_visits;
					}
				);
				move = (*it).first;
			} else {
				// Pick the edge that has the current highest k-armed bandit value.
				move = node->select_action();
			}
			// If the tree doesn't continue in the direction of this move, then break.
			const auto it = node->outgoing_edges.find(move);
			if (it == node->outgoing_edges.end())
				break;
			MCTSEdge& edge = (*it).second;
			edges_on_path.push_back(&edge);
			node = edge.child_node;
		}
		return {node, move, edges_on_path};
	}

	void step() {
		// 1) Pick a path through the tree.
		auto triple = select_principal_variation();
		// Darn, I wish I had structured bindings already. :(
		shared_ptr<MCTSNode>         leaf_node     = std::get<0>(triple);
		ChompMove                         move          = std::get<1>(triple);
		std::vector<MCTSEdge*>       edges_on_path = std::get<2>(triple);

		shared_ptr<MCTSNode> new_node;

		// 2) If the move is non-null then expand once at the leaf.
		if (move != NO_MOVE) {
			ChompState new_board = leaf_node->board;
			new_board.apply_move(move);
			new_node = std::make_shared<MCTSNode>(new_board);
			auto pair_it_success = leaf_node->outgoing_edges.insert({
				move,
				MCTSEdge{move, leaf_node.get(), new_node},
			});
			MCTSEdge& new_edge = (*pair_it_success.first).second;
			edges_on_path.push_back(&new_edge);
		} else {
			// If the move is null, then we had no legal moves, and we just propagate the score again.
			// This occurs when we're repeatedly hitting a scored final position.
			new_node = leaf_node;
		}

		// 3) Evaluate the new node to get a score to propagate back up the tree.
		new_node->populate_evals(thread_id);
		// Convert the expected value result into a score.
		double value_score = (new_node->evals.value + 1.0) / 2.0;
		// 4) Backup.
		bool inverted = false;
		for (auto it = edges_on_path.rbegin(); it != edges_on_path.rend(); it++) {
			MCTSEdge& edge = **it;
			inverted = not inverted;
			value_score = 1.0 - value_score;
			assert(inverted == (edge.parent_node->board.to_move != new_node->board.to_move));
			edge.adjust_score(value_score);
			edge.parent_node->all_edge_visits++;
		}
		if (edges_on_path.size() == 0) {
			cout << ">>> No edges on path!" << endl;
			cout << root_board;
			cout << "Valid moves list:" << endl;
			print_moves(root_board);
			cout << "Select action at root:" << endl;
			ChompMove selected_action = root_node->select_action();
			cout << "Posterior length:" << root_node->evals.posterior.size() << endl;
			cout << "Game over:" << root_node->evals.game_over << endl;
			cout << "Here it is: " << selected_action << endl;
			cout << ">>> Move:" << move << endl;
		}
		assert(edges_on_path.size() != 0);
	}

	void play(ChompMove move) {
		// See if the move is in our root node's outgoing edges.
		auto it = root_node->outgoing_edges.find(move);
		// If we miss, just throw away everything and init from scratch.
		if (it == root_node->outgoing_edges.end()) {
			root_board.apply_move(move);
			init_from_scratch(root_board);
			return;
		}
		// Otherwise, reuse a subtree.
		root_node = (*it).second.child_node;
		root_board = root_node->board;
		// Now that a new node is the root we have to redo its evals with Dirichlet noise, if required.
		// This is a little wasteful, when we could just apply Dirichlet noise, but it's not that bad.
		root_node->evals_populated = false;
		root_node->populate_evals(thread_id, use_dirichlet_noise);

	}
};

ChompMove sample_proportionally_to_visits(const shared_ptr<MCTSNode>& node) {
	double x = std::uniform_real_distribution<float>{0, 1}(generator);
	for (const std::pair<ChompMove, MCTSEdge>& p : node->outgoing_edges) {
		double weight = p.second.edge_visits / node->all_edge_visits;
		if (x <= weight)
			return p.first;
		x -= weight;
	}
	// If we somehow slipped through then return some arbitrary element.
	std::cerr << "Potential bug: Weird numerical edge case in sampling!" << endl;
	return (*node->outgoing_edges.begin()).first;
}

ChompState make_initial_board() {
	// For now always just be BOARD_SIZE x BOARD_SIZE and fully empty.
	// Later I might want to randomly cut off some rows and columns at the beginning.
	return make_empty_state();
}

std::string move_for_json(const ChompMove& m) {
	return std::to_string(m.x) + "," + std::to_string(m.y);
}

json generate_game(int thread_id) {
	ChompState board = make_initial_board();
	MCTS mcts(thread_id, board, true);
	json entry = {{"boards", {}}, {"moves", {}}, {"dists", {}}};
	int steps_done = 0;

#ifdef ONE_RANDOM_MOVE
	int random_ply = std::uniform_int_distribution<int>{0, 119}(generator);
	entry["random_ply"] = random_ply;
#endif

	for (unsigned int ply = 0; ply < maximum_game_plies; ply++) {
		// Do a number of steps.
		while (mcts.root_node->all_edge_visits < global_visits) {
			mcts.step();
			steps_done++;
		}
		// Sample a move according to visit counts.
		ChompMove selected_move = sample_proportionally_to_visits(mcts.root_node);

#ifdef ONE_RANDOM_MOVE
		// If we're AT the randomization point then instead pick a uniformly random legal move.
		if (ply == random_ply) {
			// Pick a random move.
			int random_index = std::uniform_int_distribution<int>{
				0,
				static_cast<int>(mcts.root_node->evals.posterior.size()) - 1,
			}(generator);
			auto it = mcts.root_node->evals.posterior.begin();
			std::advance(it, random_index);
			selected_move = (*it).first;
		}

		// If we're AFTER the randomization point then pick the best move we found.
		if (ply > random_ply) {
			int most_visits = -1;
			for (const std::pair<ChompMove, MCTSEdge>& p : mcts.root_node->outgoing_edges) {
				if (p.second.edge_visits > most_visits) {
					selected_move = p.first;
					most_visits = p.second.edge_visits;
				}
			}
		}
#endif

/*
		// If appropriate choose a uniformly random legal move in the opening.
		if (ply < opening_randomization_schedule.size() and
		    std::uniform_real_distribution<double>{0, 1}(generator) < opening_randomization_schedule[ply]) {
			// Pick a random move.
			int random_index = std::uniform_int_distribution<int>{0, static_cast<int>(mcts.root_node->evals.posterior.size()) - 1}(generator);
			auto it = mcts.root_node->evals.posterior.begin();
			std::advance(it, random_index);
			selected_move = (*it).first;
		}
*/
		entry["boards"].push_back(serialize_board_for_json(mcts.root_board));
		entry["moves"].push_back(move_for_json(selected_move));
		entry["dists"].push_back({});
		// Write out the entire visit distribution.
		for (const std::pair<ChompMove, MCTSEdge>& p : mcts.root_node->outgoing_edges) {
			double weight = p.second.edge_visits / mcts.root_node->all_edge_visits;
			entry["dists"].back()[move_for_json(p.first)] = weight;
		}
		mcts.play(selected_move);
		if (get_board_result(mcts.root_node->board) != 0)
			break;
	}
//	float work_factor = steps_done / (float)(global_visits * entry["moves"].size());
//	cout << "Work factor: " << work_factor << endl;

	entry["result"] = get_board_result(mcts.root_node->board);
	return entry;
}

// ================================================
//        T h r e a d e d   W o r k l o a d
// ================================================

struct Worker;
struct ResponseSlot;

std::list<Worker> global_workers;
std::vector<Worker*> global_workers_by_id;
float* global_fill_buffers[2];
int global_buffer_entries;
std::ofstream* global_output_file;

int current_buffer = 0;
int fill_levels[2] = {0, 0};
std::vector<ResponseSlot> response_slots[2];
std::queue<int> global_filled_queue;
std::mutex global_mutex;
std::atomic<bool> keep_working;

struct ResponseSlot {
	int thread_id;
};

struct Worker {
	std::mutex thread_mutex;
	std::condition_variable cv;
	std::thread t;

	bool response_filled;
	double response_value;
	float response_posterior[POSTERIOR_SIZE];

	Worker(int thread_id)
		: t(Worker::thread_main, thread_id) {}

	static void thread_main(int thread_id) {
		while (true) {
			json game;
			try {
				game = generate_game(thread_id);
			} catch (StopWorking& e) {
				return;
			}
			if (game["result"] == 0) {
				cout << "Skipping game with null result." << endl;
				continue;
			}
#ifdef ONE_RANDOM_MOVE
			if (game["random_ply"].get<int>() + 1 >= game["moves"].size()) {
				cout << "Skipping game with no board state just after the uniformly random move." << endl;
				continue;
			}
#endif
			{
				std::lock_guard<std::mutex> global_lock(global_mutex);
				cout << thread_id << " Game generated. Plies: " << game["moves"].size() << endl;
				(*global_output_file) << game << "\n";
				global_output_file->flush();
			}
		}
	}
};

std::pair<const float*, double> request_evaluation(int thread_id, const float* feature_string) {
	// Write an entry into the appropriate work queue.
	{
		std::lock_guard<std::mutex> global_lock(global_mutex);
		int slot_index = fill_levels[current_buffer]++;
		assert(0 <= slot_index and slot_index < response_slots[0].size());
		// Copy our features into the big buffer.
		float* destination = global_fill_buffers[current_buffer] + FEATURES_SIZE * slot_index;
		std::copy(feature_string, feature_string + FEATURES_SIZE, destination);
		// Place an entry requesting a reply.
		response_slots[current_buffer].at(slot_index).thread_id = thread_id;
		// Set that we're waiting on a response.
		Worker& worker = *global_workers_by_id[thread_id];
		worker.response_filled = false;
		// Swap buffers if we filled up the current one.
		if (fill_levels[current_buffer] == global_buffer_entries) {
			global_filled_queue.push(current_buffer);
			current_buffer = 1 - current_buffer;
		}
		// TODO: Notify the main thread so it doesn't have to poll.
	}
	// Wait on a reply.
	Worker& worker = *global_workers_by_id[thread_id];
	std::unique_lock<std::mutex> lk(worker.thread_mutex);
	while (not worker.response_filled) {
		worker.cv.wait_for(lk, std::chrono::milliseconds(250), [&worker]{
			return worker.response_filled;
		});
		if (not keep_working)
			throw StopWorking();
	}
	// Response collected!
	return {worker.response_posterior, worker.response_value};
}

extern "C" void launch_threads(char* output_path, int visits, float* fill_buffer1, float* fill_buffer2, int buffer_entries, int thread_count) {
	global_visits = visits;
	global_fill_buffers[0] = fill_buffer1;
	global_fill_buffers[1] = fill_buffer2;
	global_buffer_entries = buffer_entries;
	cout << "Launching into " << fill_buffer1 << ", " << fill_buffer2 << " with " << buffer_entries << " entries and " << thread_count << " threads." << endl;

	cout << "Writing to: " << output_path << endl;
	global_output_file = new std::ofstream(output_path, std::ios_base::app);
	keep_working = true;

	for (int i = 0; i < buffer_entries; i++) {
		response_slots[0].push_back(ResponseSlot());
		response_slots[1].push_back(ResponseSlot());
	}

	{
		std::lock_guard<std::mutex> global_lock(global_mutex);
		for (int i = 0; i < thread_count; i++) {
			global_workers.emplace_back(i);
			global_workers_by_id.push_back(&global_workers.back());
		}
	}
}

extern "C" int get_workload(void) {
	while (true) {
		{
			std::lock_guard<std::mutex> global_lock(global_mutex);
			// Check if a workload is ready.
			if (not global_filled_queue.empty()) {
				int workload_index = global_filled_queue.front();
				global_filled_queue.pop();
				return workload_index;
			}
		}
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	}
}

extern "C" void complete_workload(int workload, float* posteriors, float* values) {
	std::lock_guard<std::mutex> global_lock(global_mutex);
	for (int i = 0; i < global_buffer_entries; i++) {
		ResponseSlot& slot = response_slots[workload].at(i);
		Worker& worker = *global_workers_by_id.at(slot.thread_id);
		worker.response_value = values[i];
		std::copy(posteriors, posteriors + POSTERIOR_SIZE, worker.response_posterior);
		posteriors += POSTERIOR_SIZE;
		{
			std::lock_guard<std::mutex> lk(worker.thread_mutex);
			worker.response_filled = true;
		}
		worker.cv.notify_one();
	}
	fill_levels[workload] = 0;
}

extern "C" void shutdown(void) {
	keep_working = false;
	for (Worker& w : global_workers)
		w.t.join();
	// Clear out data structures to setup for another run.
	for (int i : {0, 1})
		response_slots[i].clear();
	global_workers.clear();
	global_workers_by_id.clear();
}

