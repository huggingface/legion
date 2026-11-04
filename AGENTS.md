# Tests

Do not add new unit or integration tests by default. Add targeted tests only when the user asks for tests or when the change has clear regression risk that is best covered by a test. You are allowed to create intermediate tests under a scratch/ or tmp/ folder during development. Prefer running relevant existing checks over expanding the test suite for small internal changes.

For documentation-only changes, do not add tests; run lightweight validation such as git diff --check.
