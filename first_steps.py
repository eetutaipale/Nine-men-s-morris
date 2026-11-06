
from dataclasses import dataclass
import random
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import DelayMove, UserInteraction, Video
import math
 
# ---------------------------------------------------------------------------
# 1. The basic README example: two random players in a window.
# ---------------------------------------------------------------------------
@dataclass
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
    legal = root.legal_moves
    Won = None
    node = tuple
    while Won == None:
        legal
    return node
def expand(root):
    return
def playout(root):
    return
def backpropagate(root, c):
    return
def mcts_move(model, player: int, iterations: int, c = float):
    root = Node(model.clone(), player)
 
    for _ in range(iterations):
        node = select(root, c)
        node = expand(node)
        result = playout(node)
        backpropagate(node, result)
 
    # Final choice: the most visited child.
    best = max(root.children, key=lambda child: child.visits)
    return best.move
    
def Test():
    env = mill.env(render_mode="human")
    env = UserInteraction(env)
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            break

        legal_moves = info["legal_moves"].tolist()
        
        if agent == "player_2":
            model = mill.transition_model(env.unwrapped)
            
            #move = mcts_move(model, player=2, iterations=300, c=1.41)
            move = random.choice(legal_moves)
            env.step(None)
            continue
        
        if agent == "player_1":
            destinations = {move[1] for move in legal_moves}
            for position in destinations:
                env.mark_position(position, (255, 215, 0, 120))  # Mark position 5 with a yellow shade.
            
            move = 0
            while move == 0:
                message = env.interact()  
                if message["type"] == "mouse_click" and message["position"] in destinations:
                    options = [m for m in legal_moves if m[1] == message["position"]]
                    move = random.choice(options)
                if message["type"] == "quit":
                    env.close()
                    return
            env.clear_markings()
            env.step(move)
    env.close()

if __name__ == "__main__":
    Test()