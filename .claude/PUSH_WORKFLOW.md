# Push workflow — use `git pushmain`, always

**The standard way to push this repo's `main` to origin is:**

```
git pushmain
```

That is it. Do **not** type `git push origin main` by hand anymore. The
single-token `pushmain` form is the standard going forward, not just an option —
the multi-word form is what caused the incidents documented below, and
muscle-memory is the thing we are deliberately retraining.

---

## What `git pushmain` is

A repo-local git alias (set in `.git/config`, so it travels with this clone):

```
git config alias.pushmain "push origin main"
```

Because the alias collapses `push origin main` into one token after `git`, there
is no multi-word tail for a stray newline, redirect, or autocomplete to split
into a junk file or a half-typed command. Works identically in **bash** and
**PowerShell** — it is a git-level alias, not a shell feature, so it is the
primary safeguard regardless of which shell you are in.

If you ever clone fresh and the alias is missing, re-add it with the line above.

---

## Shell convenience wrappers (secondary)

These just call the same alias so the habit is identical everywhere.

### bash (this environment — `~/.bashrc`)

```bash
gpm() { git pushmain; }
```

### PowerShell (`$PROFILE`)

```powershell
function gpm { git pushmain }
```

The git alias is the real guard; these wrappers exist so `gpm` works as
muscle-memory in either shell and still routes through the one-token path.

---

## Why this exists — the bug it prevents

Twice during this project a botched push generated a junk commit titled `h`
that reached origin:

- **`cfff008`** — cleaned up a stray tracked file literally named `h origin main`
  (30 lines). A redirect / word-split of `git pus‹split›h origin main`.
- **`a011fbf`** — a commit titled `h` that *deleted*
  `.claude/next-session-plan.md`. The file was later restored forward; history
  was **not** rewritten because the commit was already on origin.

Both map to the same failure: the multi-word `push origin main` tail getting
split by a stray newline or terminal autocomplete. Collapsing it to a single
token (`git pushmain`) removes that failure surface entirely.

**Do not** attempt to rewrite the existing `h` commits out of history — they are
already on the shared remote, and rewriting shared history is worse than the
scar.
