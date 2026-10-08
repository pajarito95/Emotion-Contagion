# Emotion-Contagion

## What's different here and other updates
- `q*` uses unweighted average (member_dynamics.py):
  $\sum_{i\self}q_i/(N-2)$; q = emotion, i = member agents, N = population size
- sparse networks allowed (network.py)
- normal signed weights implemented [-1, 1] instead of prior uniform [0, 1] (absolute value row normalization now used) (network.py)
- agents now share the same fixed parameter values of susceptibility, expressiveness, amplification, and bias (agents.py)
- network generation handling was revamped (network.py)
- `adaptive_intimacy` dictates whether to allow ties to evolve or not (default.yaml)
- emotion values reverted back to [0, 1] domain/range (member_dynamics.py)
- core-periphery generation refined (network.py)
- community structure modified to ensure leader has at least one connection to the other community (network.py)
- leader-member ties are of equal weight (1.0) and have a fixed count per periphery (network.py) 
- leaders per member influence decreases with larger N due to row-normalization (network.py)

## Considerations or To-Do:
- Add option for fixed density (network.py)
- Simplify gamma variable in `emotion_update()`? (member_dynamics.py)
- Discuss core-periphery configurations
