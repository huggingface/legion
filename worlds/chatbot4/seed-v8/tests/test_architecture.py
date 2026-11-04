import unittest

from bot import runtime
from bot.compose import pick
from bot.memory import user
from bot.understanding import understand


class ArchitectureTest(unittest.TestCase):
    def test_understanding_reads_the_kind_of_line(self):
        kinds = {"": "empty", "hey there": "greeting", "Good morning!": "greeting",
                 "thx so much": "thanks", "can u help me": "question", "Is it far?": "question",
                 "I just got a puppy": "statement"}
        for text, kind in kinds.items():
            self.assertEqual(understand(text, {})["kind"], kind, text)

    def test_every_module_sees_this_turns_understanding(self):
        memory = {}
        runtime.turn("Where is Lisbon?", memory)
        self.assertEqual(memory["understood"]["kind"], "question")
        self.assertEqual(memory["understood"]["words"], ["where", "is", "lisbon"])

    def test_memory_and_compose_helpers(self):
        memory = {}
        user(memory)["name"] = "Sam"
        self.assertEqual(memory["user"], {"name": "Sam"})
        memory["turns"] = [{"user": "hi", "bot": "Hi!", "module": "greeting"}]
        self.assertEqual(pick(["Hi!", "Hello!"], memory), "Hello!")


if __name__ == "__main__":
    unittest.main()
