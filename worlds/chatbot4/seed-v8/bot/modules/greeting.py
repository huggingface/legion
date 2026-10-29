"""The seed's fallback: it only answers when no other module does."""
PRIORITY = -1000


def respond(text, memory):
    return "Hello!"
