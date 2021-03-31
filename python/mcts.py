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
import tensorflow as tf
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

if name_scope is not None:
  with tf.compat.v1.variable_scope(name_scope):
    model = Model(model_config,pos_len,{})
else:
  model = Model(model_config,pos_len,{})
policy0_output = tf.nn.softmax(model.policy_output[:,:,0])
value_output = tf.nn.softmax(model.value_output)

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
              self.child_U - 1000 * self.illegal_moves)

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
      output.append("info move {!s:} visits {:d} utility {:.3f} winrate {:.3f} prior {:.3f} order {:d} pv {!s:} ".format(
      to_gtp(from_flat(i)),
      int(self.child_N[i]),
      round(self.child_action_score[i],6),
      round((1+self.child_Q[i])/2,6),
      round(self.child_prior[i],6),
      order,
      ' '.join(pvs[i])))
    order += 1

    return ''.join(output)


def tree_search(session, game_state, rules, fetches, num_reads):
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
    move_prob, value = NeuralNet.evaluate(session,leaf.game_state,rules,fetches)
    leaf.revert_virtual_loss(up_to=root)
    leaf.incorporate_results(move_prob, value, up_to=root)

  return root.best_child()


class Analysis():
  def __init__(self, session, game_state, rules, fetches, report_search_interval=1):
    self.session = session
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
      move_prob, value = NeuralNet.evaluate(self.session,leaf.game_state,self.rules,self.fetches)
      #leaf.add_virtual_loss(up_to=self.root)
      #move_prob, value = NeuralNet.evaluate(session,game_state,rules,fetches)
      leaf.revert_virtual_loss(up_to=self.root)
      leaf.incorporate_results(move_prob, value, up_to=self.root)
      
      if self.report_search_interval:
        now = time.time()
      if (self.last_report_time is None or now - self.last_report_time > self.report_search_interval):
        print(self.root.describe())
        self.last_report_time = time.time()


class NeuralNet():
  @classmethod
  def evaluate(self, session, gs, rules, fetches):
    bin_input_data = np.zeros(shape=[1]+model.bin_input_shape, dtype=np.float32)
    global_input_data = np.zeros(shape=[1]+model.global_input_shape, dtype=np.float32)
    pla = gs.board.pla
    opp = Board.get_opp(pla)
    move_idx = len(gs.moves)
    model.fill_row_features(gs.board,pla,opp,gs.boards,gs.moves,move_idx,rules,bin_input_data,global_input_data,idx=0)
    outputs = session.run(fetches, feed_dict={
      model.bin_inputs: bin_input_data,
      model.global_inputs: global_input_data,
      model.symmetries: [False,False,False],
      model.include_history: [[1.0,1.0,1.0,1.0,1.0]]
    })
    policy = outputs[0][0]
    #if gs.board.pla == Board.BLACK:
    value = outputs[1][0][0] - outputs[1][0][1]
    return policy, value

def get_outputs(session, gs, rules, num_reads):
  [policy0, value] = NeuralNet.evaluate(session,gs,rules,[
    policy0_output,value_output])
  board = gs.board


  result = tree_search(session, gs, rules, [policy0_output,value_output], num_reads)
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


def get_layer_values(session, gs, rules, layer, channel):
  board = gs.board
  [layer] = NeuralNet.evaluate(session,gs,rules=rules,fetches=[layer])
  layer = layer.reshape([model.pos_len * model.pos_len,-1])
  locs_and_values = []
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      locs_and_values.append((loc,layer[pos,channel]))
  return locs_and_values

def get_input_feature(gs, rules, feature_idx):
  board = gs.board
  bin_input_data = np.zeros(shape=[1]+model.bin_input_shape, dtype=np.float32)
  global_input_data = np.zeros(shape=[1]+model.global_input_shape, dtype=np.float32)
  pla = board.pla
  opp = Board.get_opp(pla)
  move_idx = len(gs.moves)
  model.fill_row_features(board,pla,opp,gs.boards,gs.moves,move_idx,rules,bin_input_data,global_input_data,idx=0)

  locs_and_values = []
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      locs_and_values.append((loc,bin_input_data[0,pos,feature_idx]))
  return locs_and_values

def get_pass_alive(board, rules):
  pla = board.pla
  opp = Board.get_opp(pla)
  area = [-1 for i in range(board.arrsize)]
  nonPassAliveStones = False
  safeBigTerritories = True
  unsafeBigTerritories = False
  board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,rules["multiStoneSuicideLegal"])

  locs_and_values = []
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      locs_and_values.append((loc,area[loc]))
  return locs_and_values


def get_gfx_commands_for_heatmap(locs_and_values, board, normalization_div, is_percent, value_and_score_from=None, hotcold=False):
  gfx_commands = []
  divisor = 1.0
  if normalization_div == "max":
    max_abs_value = max(abs(value) for (loc,value) in locs_and_values)
    divisor = max(0.0000000001,max_abs_value) #avoid divide by zero
  elif normalization_div is not None:
    divisor = normalization_div

  #Caps value at 1.0, using an asymptotic curve
  def loose_cap(x):
    def transformed_softplus(x):
      return -math.log(math.exp(-(x-1.0)*8.0)+1.0)/8.0+1.0
    base = transformed_softplus(0.0)
    return (transformed_softplus(x) - base) / (1.0 - base)

  #Softly curves a value so that it ramps up faster than linear in that range
  def soft_curve(x,x0,x1):
    p = (x-x0)/(x1-x0)
    def curve(p):
      return math.sqrt(p+0.16)-0.4
    p = curve(p) / curve(1.0)
    return x0 + p * (x1-x0)

  if hotcold:
    for (loc,value) in locs_and_values:
      if loc != Board.PASS_LOC:
        value = value / divisor

        if value < 0:
          value = -loose_cap(-value)
        else:
          value = loose_cap(value)

        interpoints = [
          (-1.00,(0,0,0)),
          (-0.85,(15,0,50)),
          (-0.60,(60,0,160)),
          (-0.35,(0,0,255)),
          (-0.15,(0,100,255)),
          ( 0.00,(115,115,115)),
          ( 0.15,(250,45,40)),
          ( 0.25,(255,55,0)),
          ( 0.60,(255,255,20)),
          ( 0.85,(255,255,128)),
          ( 1.00,(255,255,255)),
        ]

        def lerp(p,y0,y1):
          return y0 + p*(y1-y0)

        i = 0
        while i < len(interpoints):
          if value <= interpoints[i][0]:
            break
          i += 1
        i -= 1

        if i < 0:
          (r,g,b) = interpoints[0][1]
        if i >= len(interpoints)-1:
          (r,g,b) = interpoints[len(interpoints)-1][1]

        p = (value - interpoints[i][0]) / (interpoints[i+1][0] - interpoints[i][0])

        (r0,g0,b0) = interpoints[i][1]
        (r1,g1,b1) = interpoints[i+1][1]
        r = lerp(p,r0,r1)
        g = lerp(p,g0,g1)
        b = lerp(p,b0,b1)

        r = ("%02x" % int(r))
        g = ("%02x" % int(g))
        b = ("%02x" % int(b))
        gfx_commands.append("COLOR #%s%s%s %s" % (r,g,b,str_coord(loc,board)))

  else:
    for (loc,value) in locs_and_values:
      if loc != Board.PASS_LOC:
        value = value / divisor
        if value < 0:
          value = -value
          huestart = 0.50
          huestop = 0.86
        else:
          huestart = -0.02
          huestop = 0.38

        value = loose_cap(value)

        def lerp(p,x0,x1,y0,y1):
          return y0 + (y1-y0) * (p-x0)/(x1-x0)

        if value <= 0.03:
          hue = huestart
          lightness = 0.00 + 0.50 * (value / 0.03)
          saturation = value / 0.03
          (r,g,b) = colorsys.hls_to_rgb((hue+1)%1, lightness, saturation)
        elif value <= 0.60:
          hue = lerp(value,0.03,0.60,huestart,huestop)
          val = 1.0
          saturation = 1.0
          (r,g,b) = colorsys.hsv_to_rgb((hue+1)%1, val, saturation)
        else:
          hue = huestop
          lightness = lerp(value,0.60,1.00,0.5,0.95)
          saturation = 1.0
          (r,g,b) = colorsys.hls_to_rgb((hue+1)%1, lightness, saturation)

        r = ("%02x" % int(r*255))
        g = ("%02x" % int(g*255))
        b = ("%02x" % int(b*255))
        gfx_commands.append("COLOR #%s%s%s %s" % (r,g,b,str_coord(loc,board)))

  locs_and_values = sorted(locs_and_values, key=lambda loc_and_value: loc_and_value[1])
  locs_and_values_rev = sorted(locs_and_values, key=lambda loc_and_value: loc_and_value[1], reverse=True)
  texts = []
  texts_rev = []
  texts_value = []
  maxlen_per_side = 1000
  if len(locs_and_values) > 0 and locs_and_values[0][1] < 0:
    maxlen_per_side = 500

    for i in range(min(len(locs_and_values),maxlen_per_side)):
      (loc,value) = locs_and_values[i]
      if is_percent:
        texts.append("%s %4.1f%%" % (str_coord(loc,board),value*100))
      else:
        texts.append("%s %.3f" % (str_coord(loc,board),value))
    texts.reverse()

  for i in range(min(len(locs_and_values_rev),maxlen_per_side)):
    (loc,value) = locs_and_values_rev[i]
    if is_percent:
      texts_rev.append("%s %4.1f%%" % (str_coord(loc,board),value*100))
    else:
      texts_rev.append("%s %.3f" % (str_coord(loc,board),value))

  if value_and_score_from is not None:
    value = value_and_score_from["value"]
    score = value_and_score_from["scoremean"]
    lead = value_and_score_from["lead"]
    vtime = value_and_score_from["vtime"]
    texts_value.append("wv %.2fc nr %.2f%% ws %.1f wl %.1f vt %.1f" % (
      100*(value[0]-value[1] if board.pla == Board.WHITE else value[1] - value[0]),
      100*value[2],
      (score if board.pla == Board.WHITE else -score),
      (lead if board.pla == Board.WHITE else -lead),
      vtime
    ))

  gfx_commands.append("TEXT " + ", ".join(texts_value + texts_rev + texts))
  return gfx_commands

def print_scorebelief(gs,outputs):
  board = gs.board
  scorebelief = outputs["scorebelief"]
  scoremean = outputs["scoremean"]
  scorestdev = outputs["scorestdev"]
  sbscale = outputs["sbscale"]

  scorebelief = list(scorebelief)
  if board.pla != Board.WHITE:
    scorebelief.reverse()
    scoremean = -scoremean

  scoredistrmid = pos_len * pos_len + Model.EXTRA_SCORE_DISTR_RADIUS
  ret = ""
  ret += "TEXT "
  ret += "SBScale: " + str(sbscale) + "\n"
  ret += "ScoreBelief: \n"
  for i in range(17,-1,-1):
    ret += "TEXT "
    ret += "%+6.1f" %(-(i*20+0.5))
    for j in range(20):
      idx = scoredistrmid-(i*20+j)-1
      ret += " %4.0f" % (scorebelief[idx] * 10000)
    ret += "\n"
  for i in range(18):
    ret += "TEXT "
    ret += "%+6.1f" %((i*20+0.5))
    for j in range(20):
      idx = scoredistrmid+(i*20+j)
      ret += " %4.0f" % (scorebelief[idx] * 10000)
    ret += "\n"

  beliefscore = 0
  beliefscoresq = 0
  beliefwin = 0
  belieftotal = 0
  for idx in range(scoredistrmid*2):
    score = idx-scoredistrmid+0.5
    if score > 0:
      beliefwin += scorebelief[idx]
    else:
      beliefwin -= scorebelief[idx]
    belieftotal += scorebelief[idx]
    beliefscore += score*scorebelief[idx]
    beliefscoresq += score*score*scorebelief[idx]

  beliefscoremean = beliefscore/belieftotal
  beliefscoremeansq = beliefscoresq/belieftotal
  beliefscorevar = max(0,beliefscoremeansq-beliefscoremean*beliefscoremean)
  beliefscorestdev = math.sqrt(beliefscorevar)

  ret += "TEXT BeliefWin: %.2fc\n" % (100*beliefwin/belieftotal)
  ret += "TEXT BeliefScoreMean: %.1f\n" % (beliefscoremean)
  ret += "TEXT BeliefScoreStdev: %.1f\n" % (beliefscorestdev)
  ret += "TEXT ScoreMean: %.1f\n" % (scoremean)
  ret += "TEXT ScoreStdev: %.1f\n" % (scorestdev)
  ret += "TEXT Value: %s\n" % (str(outputs["value"]))
  ret += "TEXT TDValue: %s\n" % (str(outputs["td_value"]))
  ret += "TEXT TDValue2: %s\n" % (str(outputs["td_value2"]))
  ret += "TEXT TDValue3: %s\n" % (str(outputs["td_value3"]
  ))
  ret += "TEXT TDScore: %s\n" % (str(outputs["td_score"]))
  ret += "TEXT Estv: %s\n" % (str(outputs["estv"]))
  ret += "TEXT Ests: %s\n" % (str(outputs["ests"]))
  return ret





# GTP Implementation -----------------------------------------------------

#Adapted from https://github.com/pasky/michi/blob/master/michi.py, which is distributed under MIT license
#https://opensource.org/licenses/MIT
def run_gtp(session):
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
    'gogui-analyze_commands',
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
  known_analyze_commands = [
    'gfx/Policy/policy',
    'gfx/Policy1/policy1',
    'gfx/LogPolicy/logpolicy',
    'gfx/Ownership/ownership',
    'gfx/Scoring/scoring',
    'gfx/FuturePos0/futurepos0',
    'gfx/FuturePos1/futurepos1',
    'gfx/Seki/seki',
    'gfx/Seki2/seki2',
    'gfx/ScoreBelief/scorebelief',
    'gfx/PassAlive/passalive',
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

  layerdict = dict(model.outputs_by_layer)
  weightdict = dict()
  for v in tf.compat.v1.trainable_variables():
    weightdict[v.name] = v

  layer_command_lookup = dict()


  def add_extra_board_size_visualizations(layer_name, layer, normalization_div):
    assert(layer.shape[1].value == board_size)
    assert(layer.shape[2].value == board_size)
    num_channels = layer.shape[3].value
    for i in range(num_channels):
      command_name = layer_name + "-" + str(i)
      command_name = command_name.replace("/",":")
      known_commands.append(command_name)
      known_analyze_commands.append("gfx/" + command_name + "/" + command_name)
      layer_command_lookup[command_name.lower()] = (layer,i,normalization_div)

  input_feature_command_lookup = dict()
  def add_input_feature_visualizations(layer_name, feature_idx, normalization_div):
    command_name = layer_name
    command_name = command_name.replace("/",":")
    known_commands.append(command_name)
    known_analyze_commands.append("gfx/" + command_name + "/" + command_name)
    input_feature_command_lookup[command_name] = (feature_idx,normalization_div)

  for i in range(model.bin_input_shape[1]):
    add_input_feature_visualizations("input-" + str(i),i, normalization_div=1)


  linear = tf.cumsum(tf.ones([19],dtype=tf.float32),axis=0,exclusive=True) / 18.0
  color_calibration = tf.stack(axis=0,values=[
    linear,
    linear*0.5,
    linear*0.2,
    linear*0.1,
    linear*0.05,
    linear*0.02,
    linear*0.01,
    -linear,
    -linear*0.5,
    -linear*0.2,
    -linear*0.1,
    -linear*0.05,
    -linear*0.02,
    -linear*0.01,
    linear*2-1,
    tf.zeros([19],dtype=tf.float32),
    linear,
    -linear,
    tf.zeros([19],dtype=tf.float32)
  ])
  add_extra_board_size_visualizations("colorcalibration", tf.reshape(color_calibration,[1,19,19,1]),normalization_div=None)



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
      report_search_interval = int(command[-1])/100
      gs = GameState(board_size, to_play=1)
      Ana = Analysis(session,gs,rules,[policy0_output,value_output],report_search_interval)
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
      gs.board.play(pla,loc)
      gs.moves.append((pla,loc))
      gs.boards.append(gs.board.copy())
    elif command[0] == "genmove":
      outputs = get_outputs(session, gs, rules, max_playouts)
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
    elif command[0] == "policy":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["moves_and_probs0"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=outputs)
      ret = "\n".join(gfx_commands)
    elif command[0] == "policy1":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["moves_and_probs1"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=outputs)
      ret = "\n".join(gfx_commands)
    elif command[0] == "logpolicy":
      outputs = get_outputs(session, gs, rules)
      moves_and_logprobs = [(move,max(0.0,4.9+math.log10(prob))) for (move,prob) in outputs["moves_and_probs0"]]
      gfx_commands = get_gfx_commands_for_heatmap(moves_and_logprobs, gs.board, normalization_div=6, is_percent=False, value_and_score_from=outputs)
      ret = "\n".join(gfx_commands)
    elif command[0] == "ownership":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["ownership_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None, hotcold=True)
      ret = "\n".join(gfx_commands)
    elif command[0] == "scoring":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["scoring_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None, hotcold=True)
      ret = "\n".join(gfx_commands)
    elif command[0] == "futurepos0":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["futurepos0_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None, hotcold=True)
      ret = "\n".join(gfx_commands)
    elif command[0] == "futurepos1":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["futurepos1_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None, hotcold=True)
      ret = "\n".join(gfx_commands)
    elif command[0] == "seki":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["seki_by_loc"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None)
      ret = "\n".join(gfx_commands)
    elif command[0] == "seki2":
      outputs = get_outputs(session, gs, rules)
      gfx_commands = get_gfx_commands_for_heatmap(outputs["seki_by_loc2"], gs.board, normalization_div=None, is_percent=True, value_and_score_from=None)
      ret = "\n".join(gfx_commands)
    elif command[0] in layer_command_lookup:
      (layer,channel,normalization_div) = layer_command_lookup[command[0]]
      locs_and_values = get_layer_values(session, gs, rules, layer, channel)
      gfx_commands = get_gfx_commands_for_heatmap(locs_and_values, gs.board, normalization_div, is_percent=False)
      ret = "\n".join(gfx_commands)

    elif command[0] in input_feature_command_lookup:
      (feature_idx,normalization_div) = input_feature_command_lookup[command[0]]
      locs_and_values = get_input_feature(gs, rules, feature_idx)
      gfx_commands = get_gfx_commands_for_heatmap(locs_and_values, gs.board, normalization_div, is_percent=False)
      ret = "\n".join(gfx_commands)

    elif command[0] == "passalive":
      locs_and_values = get_pass_alive(gs.board, rules)
      gfx_commands = get_gfx_commands_for_heatmap(locs_and_values, gs.board, normalization_div=None, is_percent=False)
      ret = "\n".join(gfx_commands)

    elif command[0] == "scorebelief":
      outputs = get_outputs(session, gs, rules)
      ret = print_scorebelief(gs,outputs)

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

saver = tf.compat.v1.train.Saver(
  max_to_keep = 10000,
  save_relative_paths = True,
)


session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.per_process_gpu_memory_fraction = 0.3
with tf.compat.v1.Session(config=session_config) as session:
  saver.restore(session, model_variables_prefix)
  run_gtp(session)
