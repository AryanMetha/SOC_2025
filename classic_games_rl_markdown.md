# Classic Games and Reinforcement Learning

*The contents of this chapter have been derived from lecture 10 in professor David Silver's series on reinforcement learning.*

## State of the Art

Classic games have long served as testbeds for artificial intelligence, featuring simple rules yet deep strategic complexity. Many classic games have been effectively solved or dominated by AI: for example, Chinook achieved perfect play in checkers, Deep Blue achieved superhuman chess play, Logistello achieved superhuman Othello play, and TD-Gammon achieved superhuman backgammon play. These milestones illustrate that appropriately designed algorithms can master a wide range of games.

### State-of-the-art AI programs in classic games

| Game | Level of Play | Representative Program |
|------|---------------|----------------------|
| Checkers | Perfect | Chinook |
| Chess | Superhuman | Deep Blue |
| Othello | Superhuman | Logistello |
| Backgammon | Superhuman | TD-Gammon |
| Scrabble | Superhuman | Maven |
| Go (19×19) | Grandmaster | MoGo, CrazyStone, Zen |
| Heads-up Limit Texas Hold'em | Superhuman | Polaris |

Reinforcement learning (RL) methods have also produced state-of-the-art game-playing agents. Self-play RL systems have achieved strong performance in many domains: for example, KnightCap and Meep (RL chess engines), Logistello (Othello), TD-Gammon (backgammon), and Maven (Scrabble) are all examples of RL-based programs reaching high levels.

### State-of-the-art reinforcement learning agents in games

| Game | Level of Play (RL) | RL Program |
|------|-------------------|------------|
| Checkers | Perfect | Chinook (RL version) |
| Chess | International Master | KnightCap, Meep |
| Othello | Superhuman | Logistello |
| Backgammon | Superhuman | TD-Gammon |
| Scrabble | Superhuman | Maven |
| Go (19×19) | Grandmaster | MoGo, CrazyStone, Zen |
| Poker (Heads-up Limit) | Superhuman | SmooCT |

These results underscore that classical planning methods (e.g. minimax search) and learning-based methods (self-play RL) have both produced human-competitive game AI. Classic games are microcosms encapsulating real-world issues, making them ideal domains for developing and understanding AI techniques.

## Game Theory

Game playing is naturally formulated as a multi-agent, sequential decision process. A key concept is the **Nash equilibrium**: a joint strategy (policy for each player) from which no player can improve unilaterally. Formally, if players fix each other's strategies π₋ᵢ, a **best response** for player i is:

π*ᵢ(π₋ᵢ) = argmax_πᵢ Vᵢ(s; πᵢ, π₋ᵢ)

the policy maximizing player i's expected return. A Nash equilibrium is a joint policy π = (π¹, ..., πⁿ) such that πᵢ = π*ᵢ(π₋ᵢ) for every player i. Equivalently, in equilibrium no player can gain by deviating from their strategy.

In two-player zero-sum games (where one player's gain is the other's loss), the concept of **minimax** arises. Let player 1 (White) maximize and player 2 (Black) minimize rewards (so R₁ + R₂ = 0). The **minimax value** of a state s is:

v*(s) = max_π¹ min_π² V(s; π¹, π²)

i.e. White chooses a policy to maximize her value, anticipating Black's minimizing response. There is a unique minimax value function in zero-sum games, and a minimax policy profile achieving these values is a Nash equilibrium. In practice, for perfect-information games (like chess or Go) the minimax solution often corresponds to optimal play.

Games can also differ in information structure. A **perfect-information** game (e.g. chess, checkers, Go) is fully observable: at each decision all players see the complete state. An **imperfect-information** game (like poker or Scrabble) has hidden information or simultaneous moves. In such games, optimal play must account for uncertainty and beliefs. Many solution methods exist that compute approximate Nash equilibria without full observability.

In perfect-information, two-player zero-sum games, searching the game tree yields minimax solutions that are Nash-optimal. In imperfect-information settings, new methods (including learning-based approaches) are needed to approach equilibrium strategies.

## Minimax Search

Minimax search is a depth-first game-tree search that computes the minimax value by exhaustive exploration (pruned by alpha-beta). The algorithm simulates play to terminal states and backs up payoffs. Each node represents a game state; leaf nodes hold rewards; internal nodes take the max or min of children depending on which player is to move.

The classic minimax algorithm was introduced by Shannon. Its time complexity is exponential in depth, but clever pruning (alpha–beta) reduces the search drastically. In practice, high-performance game programs augment search with heuristics: for instance, Deep Blue used a deep alpha-beta search with ~8000 handcrafted board features. Similarly, Chinook (the checkers program) employed a deep search with a linear evaluation function over board features. These features capture domain knowledge (piece counts, positions, etc.) to estimate value at leaf nodes.

Formally, minimax search with depth cutoff uses a heuristic value function v(s) to evaluate non-terminal states. The minimax backup is:

v⁺(s) = {
- v(s) if s is a leaf or depth limit,
- max_a v⁺(s') if White to move at s,
- min_a v⁺(s') if Black to move at s,
}

where s' = Result(s,a) is the successor state. The algorithm composes the heuristic values via min and max operations to approximate the true minimax outcome.

While minimax search is effective, it can only look ahead a finite depth. To address this, game programs use hand-crafted or learned heuristics (evaluation functions) and search enhancements. Chinook used an endgame database plus a linear function over board features. These domain insights complement search to achieve strong play.

## Self-Play Reinforcement Learning

Reinforcement learning (RL) enables agents to learn game-playing policies by trial and error, often via self-play. In a self-play setting, multiple copies of the agent play games against each other, generating experience (Sₜ, Aₜ, Rₜ₊₁, Sₜ₊₁). In zero-sum games, self-play learning tends to drive the population towards a Nash equilibrium policy: each agent learns a best response to the current opponent, and all improve iteratively until no deviation is profitable.

A cornerstone RL algorithm is **Temporal Difference (TD) learning**, which updates a value function from successive states. For example, TD(0) updates a state-value estimate V(s) via:

V(sₜ) ← V(sₜ) + α(rₜ₊₁ + γV(sₜ₊₁) - V(sₜ))

where α is a step size, γ is a discount factor, and rₜ₊₁ is the reward.

A classic success of self-play TD learning is **TD-Gammon**. TD-Gammon learned to play backgammon solely by playing against itself and tuning a neural-network evaluation function. Its performance reached near world-champion level, validating the power of TD learning.

Monte Carlo methods are also natural in game play: one can simulate complete games and update at the end according to the return. In perfect-information games, combining Monte Carlo evaluation with heuristic search (Monte Carlo Tree Search, MCTS) has been especially successful. The UCT algorithm uses upper-confidence-bound selection to balance exploration and exploitation in the tree. Convergence results show that UCT will find minimax values with enough simulations in two-player zero-sum games.

Practically, MCTS powered breakthroughs in Go and other games by efficiently using simulations. UCT (and variants) allow an RL agent to plan lookahead using randomness. Empirically, pure MCTS-based self-play has reached top levels in Go and other games, and serves as the planning component in many modern game AIs.

## Combining Reinforcement Learning and Minimax Search

The most powerful game agents typically combine learning and search. One paradigm is to use self-play RL to train evaluation functions or policies, then incorporate them into a search. If v(s,**w**) is a parametric value estimate, one can update:

v(sₜ,**w**) ← v(sₜ,**w**) + α(v⁺(sₜ,**w**) - v(sₜ,**w**))

where v⁺(sₜ,**w**) = max_{s∈leaves(sₜ)} v(s,**w**) is the minimax-backed value at sₜ. This approach (sometimes called "TD-Search" or "TreeStrap") was used in chess and checkers to tune evaluation parameters by self-play.

Another approach is to use learned policies to guide the search. For instance, deep neural networks trained via policy gradients or self-play can serve as strong move priors. In some systems, the search algorithm combines rollouts with evaluations from the value network and move probabilities from the policy network, yielding a powerful hybrid.

A general form of the combined update (TD-Search) is:

**w** ← **w** + α(rₜ₊₁ + γv(sₜ₊₁,**w**) - v(sₜ,**w**))∇_**w**v(sₜ,**w**)

embedding the search-backed target into a gradient update for the value network.

Self-play learning can even replace exhaustive search altogether. This style of "learning to search" shows that RL and search can be tightly integrated: the search informs the learning targets, and the learned models speed up future search.

## Reinforcement Learning in Imperfect-Information Games

Imperfect-information games (such as poker) pose additional challenges. Standard minimax search is not directly applicable because players have private information. The game tree is replaced by an information-state tree: nodes represent a player's information set rather than a fully observed state.

In an information set, multiple game states (histories) are indistinguishable to the player given their observations. Many real states may share the same information state, which aggregates hidden or uncertain elements.

Traditional game-theoretic approaches (such as Counterfactual Regret Minimization) solve large games by iterative self-play to converge to an equilibrium. These methods guarantee converging to a Nash equilibrium strategy in zero-sum games, given enough iterations.

Self-play reinforcement learning has also been adapted to imperfect games. One extension is **Smooth UCT**, a variant of MCTS for information games. Smooth UCT alternates between UCT-based exploration and using the empirical mixed strategy from previous simulations. In practice, this helps the search converge to equilibrium play. Experiments in poker showed that naive MCTS diverges, whereas Smooth UCT converges towards a Nash equilibrium.

## Conclusions

Classic games provide a fertile ground for developing and evaluating reinforcement learning methods. They allow controlled study of planning, learning, and multi-agent interaction. In perfect-information games, minimax search guarantees an optimal Nash equilibrium strategy, while self-play RL can often discover the same equilibria by repeated play. Combining both paradigms leads to the strongest players: RL-trained evaluators plus deep search have achieved superhuman performance. In imperfect-information games, equilibrium concepts still guide design, and modern RL methods have pushed the frontier in poker and beyond.

In summary, the interplay between reinforcement learning and classical game search has been a key theme in advancing AI. Nash equilibrium and minimax provide theoretical foundations, while practical successes such as TD-Gammon, Logistello, Chinook, and hybrid search-learning systems demonstrate the power of self-play learning and search hybridization.