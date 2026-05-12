import unittest

from markov_chain import ChineseNameMarkov


class MarkovChainTests(unittest.TestCase):
    def test_generate_respects_seed_surname(self):
        model = ChineseNameMarkov(surnames_list=["李", "王"])
        model.train(["白", "安石", "清照"])

        generated = model.generate(seed_surname="欧阳")

        self.assertTrue(generated.startswith("欧阳"))
        self.assertGreaterEqual(len(generated), 3)


if __name__ == "__main__":
    unittest.main()
