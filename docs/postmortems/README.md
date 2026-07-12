# postmortems/

Blameless mini-postmortems, written **after a costly surprise** — a statistical bug that
shipped a wrong headline number, a false-positive screen that got confirmed too late, a
misidentified effect, or a result that didn't reproduce.

One file per incident: `YYYY-MM-DD-slug.md`, with:
- **Summary / impact** — what went wrong and what it affected.
- **Timeline** — timestamped sequence of what happened.
- **Contributing factors** — factual, evidence-backed (not blame).
- **Lessons learned** — what we now know.
- **Action items** — concrete follow-up (e.g. promote a check into a hook, add a rule to
  `stats-gatekeeper`).

The goal is to turn each surprise into a durable lesson instead of a forgotten scar.
