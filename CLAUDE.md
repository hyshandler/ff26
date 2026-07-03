# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is not a software project with source code, builds, or tests — it's a workspace configured with
Matt Pocock's Claude Code skills (`.agents/skills/`, tracked via `skills-lock.json`, sourced from
`mattpocock/skills` on GitHub). There is no build, lint, or test tooling here; don't invent any.

The `ff/` directory is currently empty.

## Working in this repo

The installed skills form a flow, routed by `/ask-matt` (see `.agents/skills/ask-matt/SKILL.md` for the
full map). Key points:

- **`/setup-matt-pocock-skills`** is the precondition — run it before any other engineering skill to
  configure the issue tracker, triage labels, and doc layout the rest assume. It has not been run yet
  in this repo (no `CONTEXT.md`, ADRs, or issue-tracker config exist).
- The main flow for new work is `/grill-with-docs` (or `/grill-me` without a codebase) → `/to-prd` →
  `/to-issues` → `/implement` (which internally drives `/tdd` and finishes with `/code-review`).
- `/triage` and `/diagnosing-bugs` are on-ramps for incoming bugs/requests and hard-to-reproduce bugs,
  respectively.
- `/handoff` bridges context across sessions; `/research` delegates background reading.

Since there's no actual code yet, the first productive step in this repo is almost always
`/setup-matt-pocock-skills`, followed by whichever flow step matches the task at hand.

## Agent skills

### Issue tracker

Issues live in GitHub Issues (hyshandler/ff26) via the `gh` CLI; external PRs are not treated as a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
