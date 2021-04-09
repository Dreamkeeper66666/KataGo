#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import re
import logging
import colorsys
import json
import paddle
from kata_paddle import *
import numpy as np
import threading
from board import Board
from model import Model
import common
import collections

description = """
Play go with a trained neural net!
Implements a basic GTP engine that uses the neural net directly to play moves.
"""

parser = argparse.ArgumentParser(description=description)
common.add_model_load_args(parser)
parser.add_argument('-name-scope', help='Name scope for model variables', required=False)
parser.add_argument('-max-playouts', help='maximum number of playouts during the search', required=False)
parser.add_argument('-max-time', help='maximum duration of the search', required=False)

args = vars(parser.parse_args())

(model_variables_prefix, model_config_json) = common.load_model_paths(args)
name_scope = args["name_scope"]
max_playouts = int(args["max_playouts"])
max_time = int(args["max_time"])

#Hardcoded max board size
pos_len = 19

# Model ----------------------------------------------------------------

with open(model_config_json) as f:
  model_config = json.load(f)
model = Model(model_config,pos_len,{})

# Basic parsing --------------------------------------------------------
colstr = 'ABCDEFGHJKLMNOPQRST'
def parse_coord(s,board):
  if s == 'pass':
    return Board.PASS_LOC
  return board.loc(colstr.index(s[0].upper()), board.size - int(s[1:]))

def str_coord(loc,board):
  if loc == Board.PASS_LOC:
    return 'pass'
  x = board.loc_x(loc)
  y = board.loc_y(loc)
  return '%c%d' % (colstr[x], board.size - y)

# Moves ----------------------------------------------------------------
def to_gtp(coord):
    """Converts from a Minigo coordinate to a GTP coordinate."""
    if coord is None:
        return 'pass'
    y, x = coord
    return '{}{}'.format(colstr[x], pos_len - y)

def from_flat(flat):
    """Converts from a flattened coordinate to a Minigo coordinate."""
    if flat == pos_len * pos_len:
        return None
    return divmod(flat, pos_len)

class GameState:
  def __init__(self, board_size, n=0, komi=7.5, to_play=1, copy_other=None):
    if copy_other is None:
      self.board_size = board_size
      self.board = Board(size=board_size)
      self.moves = []
      self.boards = [self.board.copy()]
      self.to_play = to_play
      self.n = n
      self.rules = {
      "koRule": "KO_POSITIONAL",
      "scoringRule": "SCORING_AREA",
      "taxRule": "TAX_NONE",
      "multiStoneSuicideLegal": True,
      "hasButton": False,
      "encorePhase": 0,
      "passWouldEndPhase": False,
      "whiteKomi": 7.5
      }
      self.komi = self.rules["whiteKomi"]
    else:
      self.board_size = copy_other.board_size
      self.board = copy_other.board.copy()
      self.moves = copy_other.moves.copy()
      self.boards = copy_other.boards.copy()
      self.to_play = copy_other.to_play
      self.n = copy_other.n
      self.rules = copy_other.rules
      self.komi = copy_other.komi


  def copy(self):
    return GameState(self.board_size,to_play=self.to_play, copy_other=self)


  def play_move(self, gtp_move):
    new_game_state = self.copy()
    loc = parse_coord(gtp_move,new_game_state.board)
    pla = new_game_state.board.pla
    new_game_state.board.play(pla, loc)
    new_game_state.moves.append((pla,loc))
    new_game_state.boards.append(new_game_state.board.copy())
    new_game_state.n += 1
    new_game_state.to_play *= -1

    return new_game_state

  def undo(self):
    if self.moves:
      self.moves.pop()
      self.boards.pop()
      self.n -= 1
      self.to_play *= -1
      self.board = self.boards[-1].copy()
      

  def score(self):
    board = self.board
    pla = board.pla
    opp = Board.get_opp(pla)
    area = [-1 for i in range(board.arrsize)]
    nonPassAliveStones = False
    safeBigTerritories = True
    unsafeBigTerritories = False
    board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,rules["multiStoneSuicideLegal"])
    return np.count_nonzero(area == Board.BLACK) - np.count_nonzero(area == Board.WHITE) - self.komi

  def is_game_over(self):
    return (len(self.moves) >= 2 and
      self.moves[-1][1] == Board.PASS_LOC and
      self.moves[-2][1] == Board.PASS_LOC)

  def result_string(self):
    score = self.score()
    if score > 0:
      return 'B+' + '%.1f' % score
    elif score < 0:
      return 'W+' + '%.1f' % abs(score)
    else:
      return 'DRAW'

  def all_legal_moves(self):
    'Returns a np.array of size pos_len**2 + 1, with 1 = legal, 0 = illegal'
    # by default, every move is legal
    legal_moves = np.zeros([pos_len, pos_len], dtype=np.int8)
    # ...unless there is already a stone there
    #legal_moves[self.board.board[loc] != Board.EMPTY] = 0

    for i in range(pos_len):
      for j in range(pos_len):
        loc = self.board.loc(i,j)
        if self.board.would_be_legal(self.board.pla,loc):
          legal_moves[i][j] = 1
    # and pass is always legal
    return np.concatenate([legal_moves.ravel(), [1]])     

  def flip_playerturn(self, mutate=False):
    game_state = self if mutate else self.copy()
    game_state.to_play *= -1
    return game_state


class DummyNode(object):
  """A fake node of a MCTS search tree.

  This node is intended to be a placeholder for the root node, which would
  otherwise have no parent node. If all nodes have parents, code becomes
  simpler."""
  def __init__(self):
    self.parent = None
    self.child_N = collections.defaultdict(float)
    self.child_W = collections.defaultdict(float)


class MCTSNode(object):
  """A node of a MCTS search tree.

  A node knows how to compute the action scores of all of its children,
  so that a decision can be made about which move to explore next. Upon
  selecting a move, the children dictionary is updated with a new node.

  position: A go.Position instance
  game_state: A game state instance
  fmove: A move (coordinate) that led to this position, a flattened coord
          (raw number between 0-N^2, with None a pass)
  parent: A parent MCTSNode.
  """
  def __init__(self, game_state, fmove=None, parent=None):
    if parent is None:
      parent = DummyNode()
    self.parent = parent
    self.fmove = fmove  # move that led to this position, as flattened coords
    #self.position = position
    self.game_state = game_state
    self.is_expanded = False
    self.losses_applied = 0  # number of virtual losses on this node
    # using child_() allows vectorized computation of action score.
    self.illegal_moves = 1 - self.game_state.all_legal_moves()
    self.child_N = np.zeros([pos_len * pos_len + 1], dtype=np.float32)
    self.child_W = np.zeros([pos_len * pos_len + 1], dtype=np.float32)
    # save a copy of the original prior before it gets mutated by d-noise.
    self.original_prior = np.zeros([pos_len * pos_len + 1], dtype=np.float32)
    self.child_prior = np.zeros([pos_len * pos_len + 1], dtype=np.float32)
    self.children = {}  # map of flattened moves to resulting MCTSNode
    self.cpuctExplorationBase = 500
    self.cpuctExploration = 0.9
    self.cpuctExplorationLog = 0.4

  def __repr__(self):
    return "<MCTSNode move=%s, N=%s, to_play=%s>" % (
          self.game_state.moves[-1][1], self.N, self.game_state.to_play)

  @property
  def child_action_score(self):
    return (self.child_Q * self.game_state.to_play +
              self.child_U)

  @property
  def child_Q(self):
    return self.child_W / (1 + self.child_N)

  @property
  def child_U(self):
    return ((self.cpuctExplorationLog * (math.log(
            (1.0 + self.N + self.cpuctExplorationBase) / self.cpuctExplorationBase)
                   + self.cpuctExploration)) * math.sqrt(max(1, self.N - 1)) *
            self.child_prior / (1 + self.child_N))

  @property
  def Q(self):
    return self.W / (1 + self.N)

  @property
  def N(self):
    return self.parent.child_N[self.fmove]

  @N.setter
  def N(self, value):
    self.parent.child_N[self.fmove] = value

  @property
  def W(self):
    return self.parent.child_W[self.fmove]

  @W.setter
  def W(self, value):
    self.parent.child_W[self.fmove] = value

  @property
  def Q_perspective(self):
    return self.Q * self.game_state.to_play

  def select_leaf(self):
    current = self
    pass_move = pos_len * pos_len
    while True:
      # if a node has never been evaluated, we have no basis to select a child.
      if not current.is_expanded:
        break
      # HACK: if last move was a pass, always investigate double-pass first
      # to avoid situations where we auto-lose by passing too early.
      if (current.game_state.moves and
          current.game_state.moves[-1][1] == Board.PASS_LOC and
              current.child_N[pass_move] == 0):
          current = current.maybe_add_child(pass_move)
          continue

      best_move = np.argmax(current.child_action_score)
      current = current.maybe_add_child(best_move)
    return current

  def maybe_add_child(self, fcoord):
    """Adds child node for fcoord if it doesn't already exist, and returns it."""
    if fcoord not in self.children:

      #new_position = self.position.play_move(
      #    coords.from_flat(fcoord))
      new_game_state = self.game_state.play_move(
        to_gtp(from_flat(fcoord)))
      self.children[fcoord] = MCTSNode(new_game_state, fmove=fcoord, parent=self)
    return self.children[fcoord]

  def add_virtual_loss(self, up_to):
    """Propagate a virtual loss up to the root node.

    Args:
        up_to: The node to propagate until. (Keep track of this! You'll
            need it to reverse the virtual loss later.)
    """
    self.losses_applied += 1
    # This is a "win" for the current node; hence a loss for its parent node
    # who will be deciding whether to investigate this node again.
    loss = self.game_state.to_play
    self.W += loss
    if self.parent is None or self is up_to:
      return
    self.parent.add_virtual_loss(up_to)

  def revert_virtual_loss(self, up_to):
    self.losses_applied -= 1
    revert = -1 * self.game_state.to_play
    self.W += revert
    if self.parent is None or self is up_to:
      return
    self.parent.revert_virtual_loss(up_to)

  def incorporate_results(self, move_probabilities, value, up_to):
    assert move_probabilities.shape == (pos_len * pos_len + 1,)
    # A finished game should not be going through this code path - should
    # directly call backup_value() on the result of the game.
    assert not self.game_state.is_game_over()

    # If a node was picked multiple times (despite vlosses), we shouldn't
    # expand it more than once.
    if self.is_expanded:
      return
    self.is_expanded = True

    # Zero out illegal moves.
    move_probs = move_probabilities * (1 - self.illegal_moves)
    scale = sum(move_probs)
    if scale > 0:
      # Re-normalize move_probabilities.
      move_probs *= 1 / scale

    self.original_prior = self.child_prior = move_probs
    # initialize child Q as current node's value, to prevent dynamics where
    # if B is winning, then B will only ever explore 1 move, because the Q
    # estimation will be so much larger than the 0 of the other moves.
    #
    # Conversely, if W is winning, then B will explore all 362 moves before
    # continuing to explore the most favorable move. This is a waste of search.
    #
    # The value seeded here acts as a prior, and gets averaged into Q calculations.
    self.child_W = np.ones([pos_len * pos_len + 1], dtype=np.float32) * value
    self.backup_value(value, up_to=up_to)

  def backup_value(self, value, up_to):
    """Propagates a value estimation up to the root node.

    Args:
        value: the value to be propagated (1 = black wins, -1 = white wins)
        up_to: the node to propagate until.
    """
    self.N += 1
    self.W += value
    if self.parent is None or self is up_to:
        return
    self.parent.backup_value(value, up_to)

  def is_done(self):
    """True if the last two moves were Pass or if the position is at a move
    greater than the max depth."""
    return self.game_state.is_game_over() or self.game_state.n >= 1800

  def inject_noise(self):
    epsilon = 1e-5
    legal_moves = (1 - self.illegal_moves) + epsilon
    a = legal_moves * ([FLAGS.dirichlet_noise_alpha] * (pos_len * pos_len + 1))
    dirichlet = np.random.dirichlet(a)
    self.child_prior = (self.child_prior * (1 - FLAGS.dirichlet_noise_weight) +
                        dirichlet * FLAGS.dirichlet_noise_weight)

  def children_as_pi(self, squash=False):
    """Returns the child visit counts as a probability distribution, pi
    If squash is true, exponentiate the probabilities by a temperature
    slightly larger than unity to encourage diversity in early play and
    hopefully to move away from 3-3s
    """
    probs = self.child_
    if squash:
      probs = probs ** .98
    sum_probs = np.sum(probs)
    if sum_probs == 0:
      return probs
    return probs / np.sum(probs)

  def best_child(self):
      # Sort by child_N tie break with action score.
    return np.argmax(self.child_N + self.child_action_score / 10000)

  def most_visited_path_nodes(self):
    node = self
    output = []
    while node.children:
      node = node.children.get(node.best_child())
      assert node is not None
      output.append(node)
    return output

  def most_visited_path(self):
    output = []
    node = self
    for node in self.most_visited_path_nodes():
      output.append("%s (%d) ==> " % (
        to_gtp(from_flat(node.fmove)), node.N))

    output.append("Q: {:.5f}\n".format(node.Q))
    return ''.join(output)

  def mvp_gg(self):
    """Returns most visited path in go-gui VAR format e.g. 'b r3 w c17..."""
    output = []
    for node in self.most_visited_path_nodes():
      if max(node.child_N) <= 1:
        break
      output.append(to_gtp(from_flat(node.fmove)))
    return ' '.join(output)

  def rank_children(self):
    ranked_children = list(range(pos_len * pos_len + 1))
    ranked_children.sort(key=lambda i: (
        self.child_N[i], self.child_action_score[i]), reverse=True)
    return ranked_children

  def child_most_visited_path(self):
    node = self
    output = {}
    ranked_children = self.rank_children()[:10]
    for move, child in node.children.items():
      if move in ranked_children:
        pv = []
        while child.children:
          pv_str = to_gtp(from_flat(child.fmove))
          child = child.children.get(child.best_child())
          assert node is not None
          pv.append(pv_str)
        output[move] = pv
    return output

  def describe(self):
    ranked_children = self.rank_children()[:10]
    pvs = self.child_most_visited_path()
    output = []

    order = 0
    for i in ranked_children:
      if self.child_N[i] == 0:
          break
      if self.game_state.board.pla==Board.WHITE:
        winrate = (1-self.child_Q[i])/2
        action_score = -self.child_action_score[i]
      else:
        winrate= (1+self.child_Q[i])/2
        action_score = self.child_action_score[i]

      output.append("info move {!s:} visits {:d} utility {:f} winrate {:f} prior {:f} order {:d} pv {!s:} ".format(
      to_gtp(from_flat(i)),
      int(self.child_N[i]),
      round(action_score,6),
      round(winrate,6),
      round(self.child_prior[i],6),
      order,
      ' '.join(pvs[i])))
      order += 1

    return ''.join(output)


def tree_search(pd_model, game_state, rules, fetches, num_reads):
  root = MCTSNode(game_state)
  import time
  tick = time.time()
  for i in range(num_reads):
    leaf = root.select_leaf()
    if leaf.is_done():
      value = 1 if leaf.game_state.score() > 0 else -1
      leaf.backup_value(value, up_to=root)
      continue
    leaf.add_virtual_loss(up_to=root)
    move_prob, value = NeuralNet.evaluate(pd_model,leaf.game_state,rules,fetches)
    leaf.revert_virtual_loss(up_to=root)
    leaf.incorporate_results(move_prob, value, up_to=root)

  return root.best_child()


class Analysis():
  def __init__(self, pd_model, game_state, rules, fetches, report_search_interval=1):
    self.pd_model = pd_model
    self.game_state = game_state
    self.root = MCTSNode(game_state)
    self.rules = rules
    self.fetches = fetches
    self.report_search_interval = report_search_interval
    self.last_report_time = None
    self.stop_analysis = False  
    
  def search(self):
    while True:
      if self.stop_analysis:
        ret = "done"
        return

      leaf = self.root.select_leaf()
      if leaf.is_done():
        value = 1 if leaf.game_state.score() > 0 else -1
        leaf.backup_value(value, up_to=self.root)
        continue
      leaf.add_virtual_loss(up_to=self.root)
      move_prob, value = NeuralNet.evaluate(self.pd_model,leaf.game_state,self.rules,self.fetches)
      leaf.revert_virtual_loss(up_to=self.root)
      leaf.incorporate_results(move_prob, value, up_to=self.root)
      
      if self.report_search_interval:
        now = time.time()
      if (self.last_report_time is None or now - self.last_report_time > self.report_search_interval):
        print(self.root.describe())
        sys.stdout.flush()
        self.last_report_time = time.time()


class NeuralNet():
  @classmethod
  def evaluate(self, pd_model, game_state, rules, fetches):
    bin_input_data = np.zeros(shape=[1]+model.bin_input_shape, dtype=np.float32)
    global_input_data = np.zeros(shape=[1]+model.global_input_shape, dtype=np.float32)
    pla = game_state.board.pla
    opp = Board.get_opp(pla)
    move_idx = len(game_state.moves)
    model.fill_row_features(game_state.board,pla,opp,game_state.boards,game_state.moves,move_idx,rules,bin_input_data,global_input_data,idx=0)
    outputs = pd_model(bin_input_data,global_input_data,[[1.0,1.0,1.0,1.0,1.0]])
    print(outputs)
    policy = outputs[0][0]
    if game_state.board.pla == Board.BLACK:
      value = outputs[1][0][0] - outputs[1][0][1]
    else:
      value = outputs[1][0][1] - outputs[1][0][0]

    return policy, value

def get_outputs(pd_model, gs, rules, num_reads):
  [policy0, value] = NeuralNet.evaluate(model,gs,rules,[
    policy0_output,value_output])
  board = gs.board


  result = tree_search(pd_model, gs, rules, [policy0_output,value_output], num_reads)
  genmove_result = model.tensor_pos_to_loc(result,board)

  moves_and_probs0 = []
  for i in range(len(policy0)):
    move = model.tensor_pos_to_loc(i,board)
    if i == len(policy0)-1:
      moves_and_probs0.append((Board.PASS_LOC,policy0[i]))
    elif board.would_be_legal(board.pla,move):
      moves_and_probs0.append((move,policy0[i]))


  moves_and_probs = sorted(moves_and_probs0, key=lambda moveandprob: moveandprob[1], reverse=True)

  return {
    "policy0": policy0,
    "moves_and_probs0": moves_and_probs0,
    "value": value,
    "genmove_result": genmove_result
  }


# GTP Implementation -----------------------------------------------------

#Adapted from https://github.com/pasky/michi/blob/master/michi.py, which is distributed under MIT license
#https://opensource.org/licenses/MIT
def run_gtp(pd_model):
  known_commands = [
    'boardsize',
    'clear_board',
    'showboard',
    'komi',
    'play',
    'genmove',
    'quit',
    'name',
    'version',
    'known_command',
    'list_commands',
    'protocol_version',
    'setrule',
    'policy',
    'policy1',
    'logpolicy',
    'ownership',
    'scoring',
    'futurepos0',
    'futurepos1',
    'seki',
    'seki2',
    'scorebelief',
    'passalive',
  ]

  board_size = 19
  gs = GameState(board_size, to_play=1)

  rules = {
    "koRule": "KO_POSITIONAL",
    "scoringRule": "SCORING_AREA",
    "taxRule": "TAX_NONE",
    "multiStoneSuicideLegal": True,
    "hasButton": False,
    "encorePhase": 0,
    "passWouldEndPhase": False,
    "whiteKomi": 7.5
  }


  while True:
    try:
      line = input().strip()
    except EOFError:
      break
    if line == '':
      continue
    command = [s.lower() for s in line.split()]
    if re.match('\d+', command[0]):
      cmdid = command[0]
      command = command[1:]
    else:
      cmdid = ''

    ret = ''

    if "analyze" in command[0]:
      print("= ")
      report_search_interval = int(command[-1])/100
      Ana = Analysis(pd_model,gs,rules,[policy0_output,value_output],report_search_interval)
      Ana.stop_analysis=False
      x = threading.Thread(target=Ana.search, args=())
      x.start()
    else:
        if 'Ana' in dir():
          Ana.stop_analysis=True
    if command[0] == "boardsize":
      if int(command[1]) > model.pos_len:
        print("Warning: Trying to set incompatible boardsize %s (!= %d)" % (command[1], N), file=sys.stderr)
        ret = None
      board_size = int(command[1])
      gs = GameState(board_size, to_play=1)
    elif command[0] == "clear_board":
      gs = GameState(board_size, to_play=1)
    elif command[0] == "showboard":
      ret = "\n" + gs.board.to_string().strip()
    elif command[0] == "komi":
      rules["whiteKomi"] = float(command[1])
    elif command[0] == "play":
      pla = (Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE)
      loc = parse_coord(command[2],gs.board)
      gs = gs.play_move(command[2])

    elif command[0] == "undo":
      gs.undo()

    elif command[0] == "genmove":
      outputs = get_outputs(pd_model, gs, rules, max_playouts)
      loc = outputs["genmove_result"]
      pla = gs.board.pla

      if len(command) > 1:
        pla = (Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE)
      gs.board.play(pla,loc)
      gs.moves.append((pla,loc))
      gs.boards.append(gs.board.copy())
      ret = str_coord(loc,gs.board)

    elif command[0] == "name":
      ret = 'KataGo'
    elif command[0] == "version":
      ret = '1.0'
    elif command[0] == "list_commands":
      ret = '\n'.join(known_commands)
    elif command[0] == "known_command":
      ret = 'true' if command[1] in known_commands else 'false'
    elif command[0] == "gogui-analyze_commands":
      ret = '\n'.join(known_analyze_commands)

    elif command[0] == "kata-get-rules":
      ret = "{"
      for key, value in rules.items():
        ret += "\"%s\":\"%s\"" % (key, value)
      ret += "}"

    elif command[0] == "setrule":
      ret = ""
      if command[1] == "korule":
        rules["koRule"] = command[2].upper()
      elif command[1] == "scoringrule":
        rules["scoringRule"] = command[2].upper()
      elif command[1] == "taxrule":
        rules["taxRule"] = command[2].upper()
      elif command[1] == "multistonesuicidelegal":
        rules["multiStoneSuicideLegal"] = (command[2].lower() == "true")
      elif command[1] == "hasbutton":
        rules["hasButton"] = (command[2].lower() == "true")
      elif command[1] == "encorephase":
        rules["encorePhase"] = int(command[2])
      elif command[1] == "passwouldendphase":
        rules["passWouldEndPhase"] = (command[2].lower() == "true")
      elif command[1] == "whitekomi" or command[1] == "komi":
        rules["whiteKomi"] = float(command[2])
      elif command[1] == "asym":
        rules["asymPowersOfTwo"] = float(command[2])
      else:
        ret = "Unknown rules setting"

    elif command[0] == "protocol_version":
      ret = '2'
    elif command[0] == "quit":
      print('=%s \n\n' % (cmdid,), end='')
      break
    elif "analyze" in command[0]:
      continue
    else:
      print('Warning: Ignoring unknown command - %s' % (line,), file=sys.stderr)
      ret = None

    if ret is not None:
      print('=%s %s\n\n' % (cmdid, ret,), end='')
    else:
      print('?%s ???\n\n' % (cmdid,), end='')
    sys.stdout.flush()



paddle.disable_static()
params = paddle.load(pd_model_path)
pd_model = TFModel()
pd_model.set_dict(params, use_structured_name=False)
pd_model.eval()

run_gtp(pd_model)
