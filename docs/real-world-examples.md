# Real-world Examples

## 1. Processing messages by background worker with `Depends()`

Here's a practical example of using `fastapi-injectable` in a background worker that processes messages.

This example demonstrates several key patterns for using dependency injection in background workers:

1. **Fresh Dependencies per Message**:
   - Each message gets a fresh set of dependencies through `_init_as_consumer()`
   - This ensures clean state for each message, similar to how FastAPI handles HTTP requests

2. **Proper Resource Management**:
   - Dependencies with cleanup needs (like database connections) are properly handled
   - Cleanup code in generators runs when `cleanup_exit_stack_of_func()` is called
   - Cache is cleared between messages to prevent memory leaks

3. **Graceful Shutdown**:
   - `setup_graceful_shutdown()` ensures resources are cleaned up on program termination
   - Handles both SIGTERM and SIGINT signals

```{literalinclude} ../example/worker/main.py
---
language: python
---
```

You can extend the example to re-using the business logic in your:
- Message queue consumers
- Batch processing jobs
- Long-running background tasks
- Any scenario where you need FastAPI-style dependency injection in a worker process
