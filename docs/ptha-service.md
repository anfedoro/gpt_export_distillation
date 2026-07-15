# PTHA Service Lifecycle

## Foreground and background modes

`ptha service run` owns the database, models, one `ArchiveSession`, and the Unix
socket in the current terminal. It is the only retrieval-service runtime.

`ptha service start` launches that same command in a detached session, redirects
stdin away from the terminal, writes output to the service log, records process
identity, and waits for full IPC readiness. It returns only when both models are
loaded and the session accepts requests.

```bash
ptha service start
ptha service status
ptha service restart
ptha service stop
```

The stdio process started by an MCP client is not a daemon. `ptha mcp serve` is
a lightweight JSON-RPC adapter that forwards only the two archive tools to this
service. It never owns models or the database.

## Readiness and state

Background metadata is stored as `service-state.json` in the platform state
directory. Schema version 1 records PID, process create time, executable,
command, database and socket paths, start timestamp, protocol version, and the
starting/ready phase. PID alone is never treated as ownership proof.
Each background launch also receives a cryptographically random internal
instance ID. Protected state and internal IPC status must agree before the
instance is considered owned. This ID is absent from normal human and MCP
output.

Status combines four observations: live IPC status, saved identity, current
process identity, and socket state. It reports `stopped`, `starting`, `ready`,
`degraded`, `stale-state`, or `unknown-process`. No signal is sent in the
`unknown-process` state.

Configuration:

```toml
[service]
startup_timeout_seconds = 120
shutdown_timeout_seconds = 30

[logging]
service_max_bytes = 10485760
service_backup_count = 3
```

CLI timeouts can be overridden with `service start --timeout`, `service stop
--timeout`, or the corresponding restart options.

## Stop and restart

Normal stop sends the internal IPC `shutdown` operation and waits for the exact
recorded process to exit. If IPC is unavailable, SIGTERM is allowed only after
PID, create time, and executable validation. `--force` permits SIGKILL only
after graceful shutdown and SIGTERM have failed against that same identity.

Restart performs a validated stop followed by a fresh start. The replacement
is not spawned until the previous process has exited and its stale socket has
been removed.

Lifecycle commands use an OS advisory lock at `service.lock`. A leftover plain
file does not block later commands because the kernel lock is released when its
owner exits.

## Logs and stale state

The default log is `service.log` under the platform log directory. Rotation is
bounded by the logging settings above. Logs contain lifecycle transitions,
model/schema identifiers, request IDs, operation names, durations, counts, and
sanitized exception classes. They do not contain query text, retrieved text,
message excerpts, or MCP argument bodies.

Dead process metadata and an unresponsive owned socket are classified as stale
and may be removed by the next lifecycle operation. If the saved PID belongs to
a different live process, PTHA preserves the evidence and asks for manual
inspection instead of signalling or deleting state.

`ptha service cleanup` removes only proven stale owned metadata and an
unresponsive owned Unix socket. It never signals a process. An unrelated live
PID causes refusal; `--force-state` still cannot remove an active socket or
signal that process.
