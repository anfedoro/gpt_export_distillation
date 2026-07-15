# PTHA Recovery

Start with:

```bash
ptha doctor
ptha service status --json
```

For stale service metadata or a dead socket, run `ptha service cleanup`.
Cleanup sends no signals. If the recorded PID belongs to another live process,
PTHA refuses; `--force-state` remains metadata-only and cannot signal it or
unlink an active socket.

For interrupted reindex, confirm the service is stopped and run `ptha reindex
--force`. This removes only the expected owned `.reindexing` clone and marker,
then starts a new complete rebuild. It never edits the active DB in place.

For interrupted import, the active DB was never replaced. Confirm with doctor,
then rerun the same import command to resume the preserved distillation,
canonical/chunk checkpoint, and committed embedding batches. Use
`--discard-failed` to explicitly remove that checkpoint and start a clean
build. Import and reindex share an OS maintenance lock, so concurrent mutation
is rejected.

Never manually remove state reported as `unknown-process` until its PID and
paths are inspected. PTHA deliberately avoids following symlinks or signalling
processes in that state.
