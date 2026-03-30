---
name: AFK run strategy - one task at a time
description: How to launch background training runs when user is AFK to prevent WSL inactivity crashes
type: feedback
---

Launch each training run as a **separate background task**, not chained with `&&` in one command. When a task completes, the notification triggers Claude to start the next one — this keeps WSL active with periodic activity.

**Why:** WSL2 crashes due to inactivity when a long-running chained command runs silently for hours with no activity from Claude. Individual tasks with notifications create regular touchpoints that prevent the inactivity timeout.

**How to apply:** When the user says "run these AFK", launch the first run as a background task. On completion notification, check results and immediately launch the next. Never chain more than one run in a single background bash command.
