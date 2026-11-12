# MongoDB Progress Migration Proposal

**Status: proposal only; no migration was executed.** No student progress, interaction, quiz, or event records were modified.

## Current evidence

The read-only Phase 1 audit identified 38 `student_progress` documents representing 38 student/skill pairs. Existing records contain legacy skill IDs such as `node_0001` and `node_0172`, skills 5-8 outside the canonical four-skill structure, and completed item/objective IDs not present in the current generated blueprint. Document fields are also inconsistent: some records omit completed items, completed objectives, learning events, quiz scores, quiz attempts, or next-objective data.

## Proposed mapping policy

| Legacy ID/category | Canonical ID | Mapping reason | Affected records | Risk | Validation | Rollback |
|---|---|---|---:|---|---|---|
| `node_0001`, `node_0051`, `node_0085`, `node_0130` skill references | `skill_01`-`skill_04` only after exact source-node ancestry verification | Match the node's approved chapter and Phase 1 canonical mapping, never name similarity | To be measured by dry run | High: ambiguous node ancestry can move progress between skills | Compare chapter ancestry, source node, student/skill pair, and before/after counts | Restore versioned document snapshot by `_id` |
| Legacy objective IDs | One of the 24 `obj_*` IDs only where `objective_mapper.py` provides an exact approved mapping | Use the explicit SEEK source-number mapping | To be measured by dry run | Critical: incorrect mapping changes learning history | Require one-to-one mapping and reject unmapped/ambiguous IDs | Restore original arrays from snapshot |
| Legacy progress item IDs | Canonical item ID only when the item is present in the approved source and has an exact source relationship | Do not infer equivalence from similar names | To be measured by dry run | Critical: completion could be falsely credited | Validate item type, objective, skill, and source node | Restore original item arrays/events |
| Legacy next-objective values | Canonical objective or `null` derived from verified completion | Recompute only in a separately approved migration | To be measured by dry run | High: stale recommendation may be replaced | Compare against canonical completion calculation | Restore original field |
| Missing optional arrays/maps | Normalized in application memory only until approval | Existing readers already normalize absent fields | 38-document sample from audit | Medium: write-time normalization can overwrite legacy shape | Read-only schema report | No database change required |

## Required dry-run procedure

1. Export an immutable, access-controlled snapshot of the affected documents, including `_id`, before changing anything.
2. Load the canonical four-skill/24-objective manifest and explicit mapper.
3. Produce candidate mappings for every distinct legacy skill, objective, item, and next-objective value.
4. Reject any candidate without an exact approved source relationship; do not guess.
5. Report affected document counts, completion counts, quiz attempts/scores, event counts, and collisions.
6. Validate that the candidate preserves every event, quiz attempt, score, and completion record.
7. Obtain explicit owner approval for the mapping and execution plan.

## Execution and rollback requirements

Execution must be a versioned, idempotent script that writes a migration marker and retains original fields until post-migration validation. It must use targeted updates, never delete/reset records, and stop on collisions. Rollback must restore the snapshot by immutable Mongo `_id` and verify document counts and hashes. This proposal does not authorize execution.
