---
name: fast_schedule flag
description: Never use --fast_schedule unless user explicitly requests it
type: feedback
---

Never pass --fast_schedule to any train_q*.py script unless the user explicitly asks for it.

**Why:** The flag changes the LR decay schedule and affects comparability of results. User wants full control over when it's used.

**How to apply:** Even when launching quick 15ep or 20ep test runs, always omit --fast_schedule unless the user says "use fast schedule" or "add --fast_schedule".
