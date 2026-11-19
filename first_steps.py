import random
import math
import time
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import UserInteraction


class Node:
    def __init__(self, model, player, parent=None, move=None):
        self.state = model
        self.parent = parent
        self.move = move        
        self.player = player         
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = model.legal_moves(player)          

def select(root, c):
    current_node = root
    while len(current_node.untried_actions) == 0 and len(current_node.children) > 0:
        winning_child = []
        for child in current_node.children:
            # Exploitation: choosing moves with higher average reward
            exploitation = child.wins / child.visits
            # Exploration: trying moves with less information
            exploration = exploitation + c * (((math.log(current_node.visits) / child.visits)) ** (1/2)) 
            winning_child.append((child, exploration))
        mx = -float("inf")        
        for i in range(0, len(winning_child)):
            if winning_child[i][1] > mx:
                best_child = winning_child[i][0]
                mx = winning_child[i][1]
        current_node = best_child
        
    return current_node

def expand(node):
    legal = node.untried_actions
    if len(legal) == 0:
        return node    
    else:
        move = legal[random.randrange(len(legal))]
        legal.remove(move)
        clone = node.state.clone()
        clone.make_move(node.player, move)
        if node.player == 2:
            new_node = Node(clone, 1, parent=node, move=move)
            node.children.append(new_node)
        else: 
            new_node = Node(clone, 2, parent=node, move=move) 
            node.children.append(new_node)
            
    return new_node

def simulation(node):
    max_moves = 200
    results = "draw"
    clone = node.state.clone()
    player = node.player
    moves_made = 0
    end_point = None
    while end_point is None:
        if clone.game_over() == True:
            if clone.get_phase(1) == "lost":
                results = 2          
            else:
                results = 1          
            end_point = True
            break                    

        if moves_made >= max_moves:
            results = "draw"
            end_point = True
            break

        if player == 1:
            legal1 = clone.legal_moves(1)
            if len(legal1) == 0:
                results = 2
                break
            move = legal1[random.randrange(len(legal1))]
            clone.make_move(1, move)
            player = 2
        else:
            legal2 = clone.legal_moves(2)
            if len(legal2) == 0:
                results = 1
                break
            move2 = legal2[random.randrange(len(legal2))]
            clone.make_move(2, move2)
            player = 1

        moves_made += 1

    return results

def backpropagate(node, results):
    player = node.player              
    while node is not None:
        node.visits += 1              
        if player == 1:
            if results == 2:
                node.wins += 1
            elif results == "draw":
                node.wins += 0.5
            player = 2               
        else:
            if results == 1:
                node.wins += 1
            elif results == "draw":
                node.wins += 0.5
            player = 1  
        node = node.parent


DIFFICULTIES = {
    "Easy":   {"iterations": 40,  "c": 1.41, "epsilon": 0.25},
    "Medium": {"iterations": 250, "c": 1.41, "epsilon": 0.0},
    "Hard":   {"iterations": 800, "c": 1.41, "epsilon": 0.0},
    "Random": {"iterations": 0,   "c": 1.41, "epsilon": 1.0},  

    "c=0":    {"iterations": 100, "c": 0.0,  "epsilon": 0.0},
    "c=0.5":  {"iterations": 100, "c": 0.5,  "epsilon": 0.0},
    "c=1.41": {"iterations": 100, "c": 1.41, "epsilon": 0.0},
    "c=2":    {"iterations": 100, "c": 2.0,  "epsilon": 0.0},
    "c=4":    {"iterations": 100, "c": 4.0,  "epsilon": 0.0},
}

def get_ai_move(model, player: int, legal_moves, difficulty: str):
    """Returns a tuple: (chosen_move, computation_time_in_seconds)."""
    params = DIFFICULTIES[difficulty]
    start_time = time.perf_counter()


    if params["epsilon"] > 0 and random.random() < params["epsilon"]:
        chosen_move = random.choice(legal_moves)
    else:
        root = Node(model.clone(), player)
        for _ in range(params["iterations"]):
            node = select(root, params["c"])
            node = expand(node)
            result = simulation(node)
            backpropagate(node, result)
            
        if not root.children:
            chosen_move = random.choice(legal_moves)
        else:
            best = max(root.children, key=lambda child: child.visits)
            chosen_move = best.move

    elapsed = time.perf_counter() - start_time
    return chosen_move, elapsed


def user_player(env, legal_moves):
    destinations = {move[1] for move in legal_moves}
    for position in destinations:
        env.mark_position(position, (255, 215, 0, 120))  

    move = 0
    while move == 0:
        message = env.interact()
        if message["type"] == "quit":
            env.clear_markings()
            return None                       
        if message["type"] == "mouse_click" and message["position"] in destinations:
            options = [m for m in legal_moves if m[1] == message["position"]]
            move = random.choice(options)

    env.clear_markings()
    return move

def play_human_vs_ai(difficulty: str):
    env = mill.env(render_mode="human")
    env = UserInteraction(env)
    env.reset()

    print(f"\nStarting match against AI ({difficulty})...")
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination:
            print(f"Game over! {agent} lost the game.")
            break
        if truncation:
            print("The game is too long (Draw)!")
            break

        legal_moves = info["legal_moves"].tolist()
        
        if agent == "player_1":
            move = user_player(env, legal_moves)
            if move is None:
                break
        else:
            model = mill.transition_model(env.unwrapped)
            move, elapsed = get_ai_move(model, player=2, legal_moves=legal_moves, difficulty=difficulty)
            print(f"AI ({difficulty}) moved in {elapsed:.3f} s")

        env.step(move)
    env.close()


def simulate_single_game(diff_p1: str, diff_p2: str, time_stats: dict):
    
    env = mill.env(render_mode=None)
    env.reset()
    winner = 0

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination:
            winner = 2 if agent == "player_1" else 1
            break
        if truncation:
            winner = 0
            break

        legal_moves = info["legal_moves"].tolist()
        if len(legal_moves) == 0:
            winner = 2 if agent == "player_1" else 1
            break

        model = mill.transition_model(env.unwrapped)

        if agent == "player_1":
            move, elapsed = get_ai_move(model, player=1, legal_moves=legal_moves, difficulty=diff_p1)
            time_stats[diff_p1]["total_time"] += elapsed
            time_stats[diff_p1]["moves"] += 1
        else:
            move, elapsed = get_ai_move(model, player=2, legal_moves=legal_moves, difficulty=diff_p2)
            time_stats[diff_p2]["total_time"] += elapsed
            time_stats[diff_p2]["moves"] += 1

        env.step(move)

    env.close()
    return winner

def run_tournament(games_per_pair: int = 20):
    levels = ["Easy", "Medium", "Hard"]
    time_stats = {lvl: {"total_time": 0.0, "moves": 0} for lvl in levels}
    matchup_results = []

    print(f"\nStarting tournament ({games_per_pair} games per pair, alternating starting player)...\n")

    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            lvl1, lvl2 = levels[i], levels[j]
            wins_1, wins_2, draws = 0, 0, 0

            for g in range(games_per_pair):
                
                if g % 2 == 0:
                    res = simulate_single_game(lvl1, lvl2, time_stats)
                    if res == 1: wins_1 += 1
                    elif res == 2: wins_2 += 1
                    else: draws += 1
                else:
                    res = simulate_single_game(lvl2, lvl1, time_stats)
                    if res == 1: wins_2 += 1
                    elif res == 2: wins_1 += 1
                    else: draws += 1

                print(f"  [{lvl1} vs {lvl2}] Completed {g + 1}/{games_per_pair}...", end="\r")

            print(f"Finished: {lvl1} vs {lvl2} -> {lvl1}: {wins_1} | {lvl2}: {wins_2} | Draws: {draws}      ")
            matchup_results.append((lvl1, lvl2, wins_1, wins_2, draws))

  
    print("\n" + "="*68)
    print("TOURNAMENT RESULTS (DIFFICULTY COMPARISON)")
    print("="*68)
    print(f"{'Matchup':<20} | {'Wins 1':<10} | {'Wins 2':<10} | {'Draws':<10}")
    print("-"*68)
    for lvl1, lvl2, w1, w2, d in matchup_results:
        duel_str = f"{lvl1} vs {lvl2}"
        print(f"{duel_str:<20} | {w1:<10} | {w2:<10} | {d:<10}")

    print("\n" + "="*68)
    print("AVERAGE COMPUTATION TIME PER MOVE")
    print("="*68)
    print(f"{'Difficulty':<12} | {'Iterations':<10} | {'Epsilon':<10} | {'Total Moves':<12} | {'Avg Time/Move':<14}")
    print("-"*68)
    for lvl in levels:
        iters = DIFFICULTIES[lvl]["iterations"]
        eps = DIFFICULTIES[lvl]["epsilon"]
        moves = time_stats[lvl]["moves"]
        avg_t = time_stats[lvl]["total_time"] / moves if moves > 0 else 0
        print(f"{lvl:<12} | {iters:<10} | {eps:<10} | {moves:<12} | {avg_t:.4f} s")
    print("="*68)


def run_matchup(lvl1: str, lvl2: str, games: int = 10):
    time_stats = {lvl1: {"total_time": 0.0, "moves": 0}, lvl2: {"total_time": 0.0, "moves": 0}}
    wins_1, wins_2, draws = 0, 0, 0
    p1_wins, p2_wins = 0, 0

    for g in range(games):
        
        if g % 2 == 0:
            res = simulate_single_game(lvl1, lvl2, time_stats)
            if res == 1: wins_1 += 1
            elif res == 2: wins_2 += 1
            else: draws += 1
        else:
            res = simulate_single_game(lvl2, lvl1, time_stats)
            if res == 1: wins_2 += 1
            elif res == 2: wins_1 += 1
            else: draws += 1

        
        if res == 1: p1_wins += 1
        elif res == 2: p2_wins += 1

        print(f"  [{lvl1} vs {lvl2}] Completed {g + 1}/{games}...", end="\r")

    print(f"Finished: {lvl1} vs {lvl2} -> {lvl1}: {wins_1} | {lvl2}: {wins_2} | Draws: {draws} | P1 won: {p1_wins} | P2 won: {p2_wins}      ")
    return wins_1, wins_2, draws, p1_wins, p2_wins

def run_part1(games: int = 10):
    print(f"\nMCTS vs Random and MCTS vs MCTS ({games} games each)...\n")
    w1, w2, d, _, _ = run_matchup("Medium", "Random", games)
    m1, m2, md, p1, p2 = run_matchup("Medium", "Medium", games)

    print("\n" + "="*68)
    print("PART 1 RESULTS")
    print("="*68)
    print(f"{'Matchup':<20} | {'Wins':<10} | {'Draws':<10} | {'Losses':<10}")
    print("-"*68)
    print(f"{'MCTS vs Random':<20} | {w1:<10} | {d:<10} | {w2:<10}")
    print(f"{'MCTS vs MCTS':<20} | {m1:<10} | {md:<10} | {m2:<10}")
    print(f"\nMCTS vs MCTS: player 1 won {p1}, player 2 won {p2}, draws {md}")
    print("="*68)

def run_c_test(games: int = 10):
    c_list = ["c=0", "c=0.5", "c=1.41", "c=2", "c=4"]
    print(f"\nTesting c values against c=1.41 ({games} games each)...\n")

    results = []
    for name in c_list:
        w1, w2, d, _, _ = run_matchup(name, "c=1.41", games)
        results.append((name, w1, w2, d))

    print("\n" + "="*68)
    print("UCT RESULTS (against c = 1.41)")
    print("="*68)
    print(f"{'c':<10} | {'Wins':<10} | {'Draws':<10} | {'Losses':<10} | {'Win %':<10}")
    print("-"*68)
    for name, w1, w2, d in results:
        win_percent = (w1 + 0.5 * d) / games * 100  
        print(f"{DIFFICULTIES[name]['c']:<10} | {w1:<10} | {d:<10} | {w2:<10} | {win_percent:.0f} %")
    print("="*68)


if __name__ == "__main__":
    print("=== NINE MEN'S MORRIS - MCTS ===")
    print("1. Play against AI - Easy")
    print("2. Play against AI - Medium")
    print("3. Play against AI - Hard")
    print("4. Run Difficulty Tournament")
    print("5. Run MCTS vs Random and MCTS vs MCTS")
    print("6. Run c value test")
    
    choice = input("\nSelect an option (1-6): ").strip()
    
    if choice == "1":
        play_human_vs_ai("Easy")
    elif choice == "2":
        play_human_vs_ai("Medium")
    elif choice == "3":
        play_human_vs_ai("Hard")
    elif choice in ("4", "5", "6"):
        num_games = input("How many games per pair (recommended 10 or 20)? ").strip()
        n = int(num_games) if num_games.isdigit() and int(num_games) > 0 else 10
        if choice == "4":
            run_tournament(games_per_pair=n)
        elif choice == "5":
            run_part1(games=n)
        else:
            run_c_test(games=n)
    else:
        print("Invalid option.")