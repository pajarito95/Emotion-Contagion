# Emotion-Contagion

## What's different here
- `q*` uses unweighted average (member_dynamics.py):
  $\sum_{i\self}q_i/(N-2)$; q = emotion, i = member agents, N = population size
- sparse networks allowed (network.py)
- `directed` parameter included to specify whether to create a directed network or not (network.py), but the undirected option is not functional
- normal signed weights implemented [-1, 1] instead of prior uniform [0, 1] (absolute value row normalization now used) (network.py)
- agents now share the same fixed parameter values of susceptibility, expressiveness, amplification, and bias (agents.py)
- network generation handling was revamped (network.py)

## Other notes
- `adaptive_intimacy` dictates whether to allow ties to evolve or not

## Considerations or To-Do:
- **To-Do:** Add option for fixed density (network.py)
- **To-Do:** If undirected functionality is required, a few minor adjustments are required (network.py)
- Leader continues to be excluded from emotional_valence_update() (member_dynamics.py)
- The leader is excluded from the emotion contagion function `emotion_valence_update()`
- gamma variable in `emotion_valence_update()` -- leave as-is, simplify, or exclude?
- QUESTION FOR MYSELF: was the "free" leader the ONLY RL leader, or did the other options also use RL?
