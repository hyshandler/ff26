# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A fantasy football draft projection model (see `CONTEXT.md` for domain language and scope) plus
the Matt Pocock Claude Code skills (`.agents/skills/`, tracked via `skills-lock.json`) used to
develop it. `/setup-matt-pocock-skills` has already been run — `CONTEXT.md`, `docs/adr/`, and the
GitHub Issues tracker are live.

**Before building or re-deriving anything, check `ASSETS.md`** — it indexes existing source
modules, notebooks, ADRs, and research docs, plus a list of known dead ends. If something's
missing from there, `git show archive/pre-reset-2026-07-16:<path>` has the pre-reset snapshot.

## Working in this repo

The installed skills form a flow, routed by `/ask-matt` (see `.agents/skills/ask-matt/SKILL.md` for the
full map):

- The main flow for new work is `/grill-with-docs` (or `/grill-me` without a codebase) → `/to-prd` →
  `/to-issues` → `/implement` (which internally drives `/tdd` and finishes with `/code-review`).
- `/triage` and `/diagnosing-bugs` are on-ramps for incoming bugs/requests and hard-to-reproduce bugs,
  respectively.
- `/handoff` bridges context across sessions; `/research` delegates background reading.

## Agent skills

### Issue tracker

Issues live in GitHub Issues (hyshandler/ff26) via the `gh` CLI; external PRs are not treated as a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
