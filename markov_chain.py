import random
from collections import defaultdict
from typing import Iterable, Optional


class ChineseNameMarkov:
    def __init__(self, surnames_list: Optional[Iterable[str]] = None, boost_compound: bool = False):
        self.surnames = list(surnames_list) if surnames_list is not None else []

        if boost_compound and self.surnames:
            unique_surnames = list(set(self.surnames))
            single = [surname for surname in unique_surnames if len(surname) == 1]
            compound = [surname for surname in unique_surnames if len(surname) >= 2]

            if compound:
                target_len = 10000
                boosted_pool = []
                for _ in range(int(target_len * 0.7)):
                    if single:
                        boosted_pool.append(random.choice(single))
                for _ in range(int(target_len * 0.3)):
                    boosted_pool.append(random.choice(compound))
                self.surnames = boosted_pool

        self.first_chars = []
        self.chain = defaultdict(list)

    def train(self, names_list: Iterable[str]) -> None:
        """Train the Markov chain from a list of given names."""
        count = 0
        for name in names_list:
            if not name:
                continue

            self.first_chars.append(name[0])

            for index in range(len(name) - 1):
                current_char = name[index]
                next_char = name[index + 1]
                self.chain[current_char].append(next_char)

            self.chain[name[-1]].append(None)
            count += 1

        if count == 0:
            raise ValueError("Cannot train Markov model with an empty name list.")

    def generate(self, max_length: int = 2, seed_surname: Optional[str] = None) -> str:
        """
        Generate a Chinese name using an optional fixed surname prompt.
        """
        if not self.first_chars:
            raise RuntimeError("Markov model has not been trained.")

        if seed_surname:
            surname = seed_surname
        elif self.surnames:
            surname = random.choice(self.surnames)
        else:
            raise RuntimeError("No surnames available for Markov generation.")

        current_char = random.choice(self.first_chars)
        given_name = current_char

        while len(given_name) < max_length:
            possible_next_chars = self.chain.get(current_char)
            if not possible_next_chars:
                break

            next_char = random.choice(possible_next_chars)
            if next_char is None:
                break

            given_name += next_char
            current_char = next_char

        return surname + given_name
