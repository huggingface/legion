import unittest

from bot import runtime


class RuntimeTest(unittest.TestCase):
    def test_every_turn_gets_a_reply_and_is_recorded(self):
        memory = {}
        for text in ("hi", "", "how are you?"):
            self.assertTrue(runtime.reply(text, memory).strip())
        self.assertEqual([t["user"] for t in memory["turns"]], ["hi", "", "how are you?"])

    def test_turn_names_the_chosen_module_and_every_candidate(self):
        t = runtime.turn("hello", {})
        self.assertIn(t["module"], [c["module"] for c in t["candidates"]])
        self.assertEqual(t["reply"], next(c["reply"] for c in t["candidates"] if c["module"] == t["module"]))
        priorities = [c["priority"] for c in t["candidates"]]
        self.assertEqual(priorities, sorted(priorities, reverse=True))


if __name__ == "__main__":
    unittest.main()
