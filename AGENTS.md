# Tests

Do not add new unit or integration tests by default. Add targeted tests only when the user asks for tests or when the change has clear regression risk that is best covered by a test. You are allowed to create intermediate tests under a scratch/ or tmp/ folder during development. Prefer running relevant existing checks over expanding the test suite for small internal changes.

For documentation-only changes, do not add tests; run lightweight validation such as git diff --check.

## Real objects

Tests exercise real objects: the resource providers in `resources/`, a real local msgboard board, real child processes and real files. Do not use `SimpleNamespace`, `MagicMock`, `object.__new__` (building an instance without its `__init__`), hand-written stand-in providers or servers, or monkeypatched internals as stand-ins: they only have what the test author thought of, so the test keeps passing when the real code changes. Mock only what cannot run in the test environment, like a remote server. Shortening a timing constant or setting an environment variable so that a real run fits in a test is fine.

Test behaviour through the public API (`Resources`, a provider's `start`, `bind`, `export` and `stop`, the CLI) rather than private methods. Call a private function directly only when its behaviour cannot be reached deterministically otherwise, such as a pure helper.

Keep the suite small: a few focused tests of core behaviour, not one test per edge case or timing constant.

## Hugging Face Spaces

Flock's only Hub calls are `msgboard.space.deploy()`, which msgboard tests and Flock runs in a child process. What only a Space does (restart hand-over, relaunch, outage) is covered by one opt-in end-to-end test, marked `live`. It is skipped unless `FLOCK_LIVE_SPACES=<namespace>/<name-prefix>` is set and a Hugging Face token is available, deploys CPU Basic Spaces, and deletes everything it creates. Its output must never show a secret: assert on computed values (statuses, booleans), never on an expression that holds a token, and let its helper turn failures into sanitized messages without a traceback. If a test ever records Hub traffic with pytest-vcr, scrub tokens from request and response bodies as well as headers (Space secrets travel in request bodies), and check every cassette for token-like strings before committing it.
