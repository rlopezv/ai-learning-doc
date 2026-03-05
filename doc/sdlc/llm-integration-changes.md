## LLM integration changes
- [ ] Are all LLM calls wrapped in retry logic with exponential backoff?
- [ ] Is token counting implemented to prevent context overflow?
- [ ] Are API errors handled gracefully (no bare `except`)?
- [ ] Is the temperature set appropriately for the task (0 for deterministic)?

---
[« Back to sdlc Index](index.md) | [🏠 Home](../../index.md)