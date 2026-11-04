import json
import unittest
from pathlib import Path

DECK = Path(__file__).resolve().parents[1] / "deck"


class DeckTest(unittest.TestCase):
    def test_conversations_are_well_formed_and_ids_unique(self):
        seen = set()
        for path in sorted(DECK.glob("*.jsonl")):
            for n, line in enumerate(path.read_text().splitlines(), 1):
                item = json.loads(line)
                where = f"{path.name}:{n}"
                self.assertEqual(set(item), {"id", "area", "turns"}, where)
                self.assertNotIn(item["id"], seen, where)
                seen.add(item["id"])
                self.assertTrue(item["area"].strip(), where)
                self.assertTrue(item["turns"], where)
                for text in item["turns"]:
                    self.assertTrue(isinstance(text, str) and text.strip() and "\n" not in text, where)
        self.assertTrue(seen)

    def test_single_and_multi_turn_files_hold_their_kind(self):
        for name, multi in (("single.jsonl", False), ("multi.jsonl", True)):
            for line in (DECK / name).read_text().splitlines():
                self.assertEqual(len(json.loads(line)["turns"]) > 1, multi, line)


if __name__ == "__main__":
    unittest.main()
