
import random
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import DelayMove, UserInteraction, Video
import math
 
# ---------------------------------------------------------------------------
# 1. The basic README example: two random players in a window.
# ---------------------------------------------------------------------------
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
    Current_node = root
    while len(Current_node.untried_actions) == 0 and len(Current_node.children) > 0:
        Winning_child = []
        for child in Current_node.children:
            #Exploitation: choosing moves with higher average reward
            Exploitation = child.wins / child.visits
            #Exploration::  trying moves with less information
            Exploration = Exploitation + c *(((math.log(Current_node.visits) / child.visits)) ** (1/2) ) 
            Winning_child.append((child, Exploration))
        mx = -float("inf")        
        for i in range(0, len(Winning_child)):
            if Winning_child[i][1] > mx:
                best_child = Winning_child[i][0]
                mx = Winning_child[i][1]
        Current_node = best_child
        
    return Current_node

def expand(node):
    legal = node.untried_actions
    if len(legal) == 0:
         return node    
    else:
        move = legal[random.randrange(len(legal))]
        legal.remove(move)
        Clone = node.state.clone()
        Clone.make_move(node.player, move)
        if node.player == 2:
            new_node = Node(Clone, 1, parent=node, move = move)
            node.children.append(new_node)
        else: 
            new_node = Node(Clone, 2, parent=node, move = move) 
            node.children.append(new_node)
            
    return new_node
def simulation(node):
    max_moves=200
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
            move = legal1[random.randrange(len(legal1))]
            clone.make_move(1, move)
            player = 2
        else:
            legal2 = clone.legal_moves(2)
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
        
def mcts_move(model, player: int, iterations: int, c = float):
    root = Node(model.clone(), player)
 
    for _ in range(iterations):
        node = select(root, c)
        node = expand(node)
        result = simulation(node)
        backpropagate(node, result)
    best = max(root.children, key=lambda child: child.visits)
    return best.move
def User_player(env, legal_moves):
    destinations = {move[1] for move in legal_moves}
    for position in destinations:
        env.mark_position(position, (255, 215, 0, 120))  # Yellow: possible destinations.

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
def Test():
    env = mill.env(render_mode="human")
    env = UserInteraction(env)
    
    env.reset()
    for agent in env.agent_iter():
        observation, reward,termination, truncation, info = env.last()
        if termination:
            print(f"{agent} lost the game!")
            break
        if truncation:
            print("The game is too long!")
            break

        legal_moves = info["legal_moves"].tolist()
        
        if agent == "player_2":
            model = mill.transition_model(env.unwrapped)
            
            move = mcts_move(model, player=2, iterations=300, c=1.41)
            env.step(move)
            continue
        
        if agent == "player_1":
            #move = User_player(env, legal_moves)
            #move = random.choice(legal_moves)
            model2 = mill.transition_model(env.unwrapped)
            move = mcts_move(model2, player=1, iterations=300, c=1.41)
            if move is None:                  
                break
        env.step(move)
    env.close()

if __name__ == "__main__":
    Test()