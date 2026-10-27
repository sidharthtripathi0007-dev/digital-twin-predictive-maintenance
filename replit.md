# TURBINE_01 Predictive Maintenance

Streamlit college-project dashboard demonstrating a simulated industrial digital twin and rule-based maintenance estimates for TURBINE_01.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/turbine-maintenance/app.py` — Streamlit dashboard and session-state history
- `artifacts/turbine-maintenance/src/` — simulated twin, RUL heuristic, maintenance rules, and network timing
- `requirements.txt` and `README.md` — Python dependencies and run instructions

## Architecture decisions

- All readings and network timings are generated in-memory; no physical sensors or external services are required.
- RUL is a transparent simulated heuristic, not output from a trained or validated deep-learning model.
- Dashboard state and history are session-local and reset when a new session starts.

## Product

Demonstrates changing Healthy, Degrading, and Critical turbine conditions, live synthetic sensor telemetry, sensor and status history, configurable RUL alerts, rule-based maintenance recommendations, and simulated network delay.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

_Populate as you build — sharp edges, "always run X before Y" rules._

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
