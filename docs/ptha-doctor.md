# PTHA Doctor

`ptha doctor` is a read-only operational audit. It does not load embedding
models or download model files. Checks have stable IDs and `PASS`, `WARN`, or
`FAIL` status; only failures produce a non-zero exit code.

The lightweight catalog covers configuration/default resolution, directories
and permissions, free space, clean-native DB layout and integrity,
schema/model/chunk metadata, canonical/derived counts, absence of legacy
fallback tables, incomplete operations, lifecycle identity, service/DB
agreement, runtime versions, and dependencies. Attachment content is reported
as a v1 limitation.

```bash
ptha doctor
ptha doctor --json
```

`ptha doctor --full` additionally validates models, embedding spaces, one real
archive session, focused retrieval, and broad context construction. If a
service is ready, checks go through it and do not load a second model copy.
Otherwise doctor loads a temporary local runtime and closes it afterward.

```bash
ptha doctor --full
ptha doctor --full --query "my local smoke query"
```

Without `--query`, a neutral technical probe is used and a positive match is
not required. Reports include timings, mode, RSS, and sanitized exception class
only. Retrieved archive text is never included.
