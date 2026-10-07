import numpy as np

from Utilities.game_state import GameState
from Utilities.shared_utils import get_high_frequency_candidates, score_patterns

class EntropyBot:
    def __init__(self, word_list: list[str], pattern_table: np.ndarray | None = None) -> None:
        """
        Args:
            word_list: The master word list
            pattern_table: Precomputed N×N pattern table. If None, patterns are scored on the fly,
                           which is plenty fast after the first guess (the opener is always "crane")
                           and avoids loading the 168 MB table at all.
        """
        self.game_state = GameState(word_list)
        self.pattern_table = pattern_table


    def calculate_entropies(self, candidates: list[str]) -> np.ndarray:
        """
        Incredibly fast entropy calculation using NumPy broadcasting and bincount.

        Args:
            candidates: The words to calculate entropy for

        Returns:
            np.ndarray: The entropy of each candidate over the remaining words

        """
        remaining = self.game_state.remaining_words

        # Patterns for each candidate against ONLY the remaining words
        if self.pattern_table is not None:
            candidate_indices = [self.game_state.word_to_index[word] for word in candidates]
            patterns = self.pattern_table[np.ix_(candidate_indices, self.game_state.remaining_words_indices)]
        else:
            patterns = score_patterns(candidates, remaining)

        # Count each (candidate, pattern) pair that actually occurs. Giving every candidate its own block
        # of 243 ids lets one np.unique count them all without allocating empty bins for unseen patterns
        keys = (patterns + np.arange(len(candidates))[:, None] * 243).ravel()
        unique_keys, counts = np.unique(keys, return_counts=True)

        # Vectorized entropy math: -sum(P * log2(P)) = log2(n) - sum(c * log2(c)) / n
        n = len(remaining)
        weighted = np.bincount(unique_keys // 243, weights=counts * np.log2(counts), minlength=len(candidates))
        return np.log2(n) - weighted / n

    def calculate_entropy(self, guess: str) -> float:
        """
        Entropy of a single guess over the remaining words.

        Args:
            guess: The string to calculate entropy for

        """
        return float(self.calculate_entropies([guess])[0])

    def make_guess(self) -> str:
        """
        Make a Guess That Maximizes Entropy in Order to Shrink the Remaining Word List as Much as Possible
        First Guess is Always "Crane" Due to it Being Optimal

        Returns:
            str: The Bot's guess for that round

        """

        if self.game_state.guess_count == 0:
            self.game_state.guess_count += 1  # Increment the GameState guess count
            return "crane"

        self.game_state.guess_count += 1  # Increment the GameState guess count
        remaining_words_length = len(self.game_state.remaining_words)

        if remaining_words_length == 1:  #No entropy calculations for just 1 word
            return self.game_state.remaining_words[0]

        #If it has a lot of possible words, it just checks the top 300 words with high-frequency letters
        if remaining_words_length > 20:
            guess_candidates = get_high_frequency_candidates(self.game_state, 300, self.game_state.master_list)
        else:
            #Below 20 remaining words is dangerous because the bot can get stuck in traps like LIGHT, MIGHT, SIGHT, etc.
            #To combat this, allow the bot to make a sacrificial guess like "MILES" to rule out LIGHT, MIGHT, and SIGHT
            guess_candidates = self.game_state.master_list

        entropies = self.calculate_entropies(guess_candidates)

        #This acts as a tiebreaker because we would PREFER to guess a word that could actually be the answer.
        #So if both are high-entropy, pick one that COULD actually be the answer.
        remaining = set(self.game_state.remaining_words)
        entropies += np.array([0.01 if word in remaining else 0.0 for word in guess_candidates])

        return guess_candidates[int(np.argmax(entropies))]  # argmax keeps the first word on ties
