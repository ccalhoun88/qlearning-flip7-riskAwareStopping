# Diante Calhoun - CIS 730 Term Project
# This is the Flip 7 engine containing the deck creation, player state and core game mechanics.

import random
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

# Adding a engine specific logger for the game events
engine_logger = logging.getLogger("flip7.engine")

def log(msg: str) -> None:
    engine_logger.info(msg)

Card = Union[int, str]

def create_flip7_deck() -> List[Card]:
    """
    Creates and shuffles a standard Flip 7 deck.
    
    Number cards:
        one 1, two 2s, three 3s, ... twelve 12s
    
    Special cards:
        freeze x3, draw3 x3, second_chance x2
        +2 x2, +4 x2, +6 x1, +8 x1, +10 x1
        x2 x1, zero x1

    Total Cards: 95 cards
    """
    deck: List[Card] = []

    # Number cards — value determines quantity
    for value in range(1, 13):
        deck.extend([value] * value)

    # Special cards - Tune as needed if you'd like to fit the true game
    deck.extend(["freeze"] * 3)
    deck.extend(["draw3"] * 3)
    deck.extend(["second_chance"] * 2)
    deck.extend(["+2"] * 2)
    deck.extend(["+4"] * 2)
    deck.extend(["+6"] * 1)
    deck.extend(["+8"] * 1)
    deck.extend(["+10"] * 1)
    deck.extend(["x2"] * 1)
    deck.extend([0] * 1)

    random.shuffle(deck)
    return deck


# Player State function
@dataclass
class PlayerState:
    """
    Tracks all state for a single player within a game.
    Reset between rounds but total_points persists across rounds.
    """
    name: str
    round_points: int = 0
    total_points: int = 0
    drawn_numbers: set = field(default_factory=set) # important since this creates a new instance each time the function is called. Key to this game.
    cards_drawn_count: int = 0
    busted: bool = False
    stopped: bool = False
    frozen: bool = False
    second_chance_available: bool = False
    multiplier: int = 1

    def is_active(self) -> bool:
        """Player can still act this round."""
        return not self.busted and not self.stopped and not self.frozen

    def reset_round(self) -> None:
        """
        Resets round-level state between rounds.
        total_points intentionally preserved — carries across rounds.
        """
        self.round_points = 0
        self.drawn_numbers = set()
        self.cards_drawn_count = 0
        self.busted = False
        self.stopped = False
        self.frozen = False
        self.second_chance_available = False
        self.multiplier = 1

# flip7 Round Engine
class Flip7RoundEngine:
    """
    Manages game state, card application, and special card resolution
    for a single Flip 7 game session.
    """
    def __init__(self, player_names: List[str], 
                 seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.players = [PlayerState(name=n) for n in player_names]
        self.deck: List[Card] = []
        self.turn_index = 0

    def reset_round(self) -> None:
        """
        Resets deck and all player round states.
        Called at the start of every new round.
        """
        self.deck = create_flip7_deck()
        self.turn_index = 0
        for player in self.players:
            player.reset_round()

      def draw_card(self) -> Card:
        """Pulls the top card from the deck."""
        if not self.deck:
            raise RuntimeError("Deck is empty.")
        return self.deck.pop()

    def apply_card_by_index(self, player_idx: int, card: Card) -> None:
        """
        Applies a drawn card to the player at player_idx.
        Handles number cards, bust logic, second chance,
        and all special card effects.
        """
        player = self.players[player_idx]
        player.cards_drawn_count += 1

        if isinstance(card, int):
            if card in player.drawn_numbers:
                if player.second_chance_available:
                    player.second_chance_available = False
                    log(f"{player.name} used Second Chance to avoid busting.")
                else:
                    player.busted = True
                    player.round_points = 0
            else:
                player.drawn_numbers.add(card)
                player.round_points += card

        elif card == "second_chance":
            player.second_chance_available = True
            log(f"{player.name} gained Second Chance.")

        elif card == "freeze":
            target_idx = self.freeze_target_heuristic(player_idx)
            if target_idx is not None:
                self.players[target_idx].frozen = True
                log(f"{player.name} froze {self.players[target_idx].name}.")

        elif card == "draw3":
            target_idx = self.draw3_target_heuristic(player_idx)
            log(f"{player.name} applies Draw 3 to {self.players[target_idx].name}.")
            self.force_draw_cards(target_idx, num_cards=3)

        elif card == "x2":
            player.multiplier *= 2
            log(f"{player.name} multiplier increased to x{player.multiplier}.")

        elif isinstance(card, str) and card.startswith("+"):
            bonus = int(card.replace("+", ""))
            player.round_points += bonus
            log(f"{player.name} gains bonus {bonus} points.")

        else:
            raise ValueError(f"Unknown card: {card}")

    def bank_points(self, player: PlayerState) -> None:
        """
        Banks player's round points into total and marks them stopped.
        Multiplier applied at banking time.
        """
        if not player.busted:
            player.total_points += player.round_points * player.multiplier
        player.stopped = True

    def all_players_done(self) -> bool:
        """Returns True when every player is busted, stopped, or frozen."""
        return all(
            p.busted or p.stopped or p.frozen 
            for p in self.players
        )
      
    def get_leading_opponent(self, current_player_idx: int) -> Optional[int]:
        """
        Returns index of opponent with highest combined total + round points.
        Used for freeze targeting and score gap calculation.
        """
        opponents = [
            (idx, p.total_points + p.round_points)
            for idx, p in enumerate(self.players)
            if idx != current_player_idx
        ]
        if not opponents:
            return None
        return max(opponents, key=lambda x: x[1])[0]
      

    def print_round_state(self) -> None:
        """Logs full state snapshot for all players."""
        for p in self.players:
            log(
                f"{p.name}: round={p.round_points}, total={p.total_points}, "
                f"drawn={sorted(p.drawn_numbers)}, busted={p.busted}, "
                f"stopped={p.stopped}, frozen={p.frozen}, "
                f"2nd={p.second_chance_available}, x{p.multiplier}"
            )

    def freeze_target_heuristic(self, current_player_idx: int) -> Optional[int]:
        """Freeze the leading opponent."""
        return self.get_leading_opponent(current_player_idx)

    def draw3_target_heuristic(self, current_player_idx: int,
                                trailing_threshold: int = 15,
                                opponent_card_threshold: int = 3) -> int:
        """
        If trailing significantly apply Draw 3 to self.
        Otherwise apply draw 3 to leading opponent if they have enough cards.
        """
        player = self.players[current_player_idx]
        leading_idx = self.get_leading_opponent(current_player_idx)

        if leading_idx is None:
            return current_player_idx

        leading_score = (self.players[leading_idx].total_points +
                        self.players[leading_idx].round_points)
        my_score = player.total_points + player.round_points
        gap = leading_score - my_score

        if gap >= trailing_threshold:
            return current_player_idx
        elif self.players[leading_idx].cards_drawn_count >= opponent_card_threshold:
            return leading_idx
        else:
            return current_player_idx

    def force_draw_cards(self, player_idx: int, num_cards: int = 3) -> None:
        """
        Forces a player to draw num_cards cards.
        Respects bust logic and second chance.
        Stops early if the player busts.
        """
        player = self.players[player_idx]
        for _ in range(num_cards):
            if not self.deck:
                break
            card = self.draw_card()
            log(f"{player.name} forced draw: {card}")
            if isinstance(card, int):
                if card in player.drawn_numbers:
                    if player.second_chance_available:
                        player.second_chance_available = False
                        log(f"{player.name} used Second Chance to avoid busting.")
                    else:
                        player.busted = True
                        player.round_points = 0
                        log(f"{player.name} busted during forced draws.")
                        break
                else:
                    player.drawn_numbers.add(card)
                    player.round_points += card
            elif card == "second_chance":
                player.second_chance_available = True
                log(f"{player.name} gained Second Chance.")
            elif card == "x2":
                player.multiplier *= 2
                log(f"{player.name} multiplier increased to x{player.multiplier}.")
            elif isinstance(card, str) and card.startswith("+"):
                bonus = int(card.replace("+", ""))
                player.round_points += bonus
                log(f"{player.name} gains bonus {bonus} points.")
            elif card == "freeze":
                target_idx = self.freeze_target_heuristic(player_idx)
                if target_idx is not None:
                    self.players[target_idx].frozen = True
                    log(f"{player.name} froze {self.players[target_idx].name}.")
            elif card == "draw3":
                target_idx = self.draw3_target_heuristic(player_idx)
                log(f"{player.name} applies Draw 3 to {self.players[target_idx].name}.")
                self.force_draw_cards(target_idx, num_cards=3)

#### Run the Round Logic ####
def run_round(engine: Flip7RoundEngine, policies: dict) -> None:
    """
    Runs one full round. Players cycle in sequence until all are done.
    policies: dict mapping player index -> callable(player, engine, idx) -> "DRAW" or "STOP"
    """
    engine.reset_round()

    while not engine.all_players_done():
        for idx, player in enumerate(engine.players):
            if not player.is_active():
                continue

            action = policies[idx](player, engine, idx)

            if action == "STOP":
                engine.bank_points(player)
            elif action == "DRAW":
                card = engine.draw_card()
                log(f"{player.name} drew: {card}")
                engine.apply_card_by_index(idx, card)
                if player.busted:
                    player.stopped = True

    # Bank frozen players at round end
    for player in engine.players:
        if player.frozen and not player.busted:
            player.total_points += player.round_points * player.multiplier

#### Run the Game logic ####
def run_game(player_names: list, policies: dict, win_threshold: int = 200,
             max_rounds: int = 50, verbose: bool = False) -> dict:
    """
    Runs a full game until one player reaches win_threshold points.
    Returns results dict with winner, final scores, and round count.
    """
    engine = Flip7RoundEngine(player_names)
    round_num = 0

    while round_num < max_rounds:
        round_num += 1
        run_round(engine, policies)

        if verbose:
            log(f"\n{'='*40}")
            log(f"  END OF ROUND {round_num} SUMMARY")
            log(f"{'='*40}")
            engine.print_round_state()
            log(f"{'='*40}\n")

        for player in engine.players:
            if player.total_points >= win_threshold:
                return {
                    "winner": player.name,
                    "scores": {p.name: p.total_points for p in engine.players},
                    "rounds": round_num
                }

    # Max rounds hit — highest score wins
    winner = max(engine.players, key=lambda p: p.total_points)
    return {
        "winner": winner.name,
        "scores": {p.name: p.total_points for p in engine.players},
        "rounds": round_num
    }


