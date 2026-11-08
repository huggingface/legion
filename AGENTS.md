# Tests

Do not add new unit or integration tests by default. Add targeted tests only when the user asks for tests or when the change has clear regression risk that is best covered by a test. You are allowed to create intermediate tests under a scratch/ or tmp/ folder during development. Prefer running relevant existing checks over expanding the test suite for small internal changes.

For documentation-only changes, do not add tests; run lightweight validation such as git diff --check.

## No mocks or fake services

Tests run real code against real services. Do not add `unittest.mock`, monkeypatched service or SDK calls, fake servers, or other stand-ins for msgboard, the Hugging Face Hub or any other service. Use a real local msgboard board where a local board is what is under test. Test behaviour that only a Hugging Face Space has (deploy, restart, relaunch, outage) against real Spaces in live tests. Live tests are marked `live`, are skipped unless `FLOCK_LIVE_SPACES=<namespace>/<name-prefix>` is set and a Hugging Face token is available, and delete everything they create. If a behaviour can't be tested without a stand-in, leave it untested and say so in the pull request.
