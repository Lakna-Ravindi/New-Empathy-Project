# SEEK Knowledge Base and Progress-Tracking Read-Only Audit

Audit scope: inspection only. The knowledge-base JSON, progress JSON, curriculum JSON, validation report, Python source, configuration, and database records were not modified. The only file created by this audit is this report.

## 1. Project file inventory

| File | Exists | Purpose and connections |
|---|---:|---|
| `output/knowledge_base.json` | Yes | 205 extracted SEEK nodes. Loaded by `server.py` for chatbot retrieval and used by `app.py` when rebuilding the extraction. |
| `output/progress_structure.json` | Yes | Version 3, generated from the knowledge base. Used by `ProgressService`, progress routes, objective completion, and next-objective selection. |
| `output/content_skills.json` | Yes | Authored curriculum-style data. Used by `server.py` for `/api/content/skills`; its objective set conflicts with the PDF-derived 24-objective structure. |
| `output/validation_report.json` | Yes | Prior generated validation output. Its counts do not match the current knowledge base metadata and should not be treated as a current audit result. |
| `backend/knowledge/objective_mapper.py` | Yes | Maps PDF objective numbering and content ancestry to canonical objective IDs. |
| `backend/knowledge/progress_builder.py` | Yes | Builds progress structures from nodes or, through a separate legacy function, from authored skill data. |
| `backend/learning/progress_service.py` | Yes | Calculates objective, skill, and overall progress and evaluates activity/quiz completion. |
| `backend/learning/pedagogical_controller.py` | Yes | Maps questions to skills/objectives, selects content and activities, and prepares Gemini input. |
| `backend/learning/interaction_store.py` | Yes | MongoDB/local-fallback persistence for interactions, progress, quiz attempts, learning events, and evaluations. |
| `backend/server.py` | Yes | Flask API. Loads the knowledge base and progress structure and exposes content, learning-response, progress, completion, and evaluation routes. |
| `backend/app.py` | Yes | PDF extraction/rebuild script, not the Flask API entry point. |

Relevant dependencies not fully audited in this report include `highlighted_keywords.json`, `keyword_skill_map.json`, classifier/parser modules, authentication/user-store code, Gemini responder implementation, registration assessment code, and frontend callers. Their references were inspected where they affected the traced routes.

## 2. Knowledge-base summary

### Current node totals

- Total nodes: **205**
- Node types:
  - `content`: 126
  - `learning_objective`: 48
  - `topic`: 17
  - `objective_heading`: 9
  - `chapter`: 4
  - `practice`: 1
- Unique node IDs: **205**
- Duplicate node IDs: **none**
- Skills represented by chapter nodes: **4**
- Learning-objective nodes: **48**, consisting of two extracted representations for each of the **24 unique objective IDs**
- Unique objective IDs: **24**

### Current objective inventory

The following is the current mapping in `knowledge_base.json`. The node ID shown is the first source-objective node for each canonical ID.

#### Skill 1 — Calming the Body and Mind

| Objective ID | Title | KB node |
|---|---|---|
| `obj_01_01` | 1. Discuss the importance of Body literacy and recognize the emotional state in a given situation. | `node_0004` |
| `obj_01_05` | 2. Identify the right vocabulary to name the different sensations in the body. | `node_0005` |
| `obj_01_02` | 3. Define the resilient zone and its connection to emotional regulation. | `node_0006` |
| `obj_01_06` | 4. Recognize when you are out of the resilience zone. | `node_0007` |
| `obj_01_03` | 5. Discuss the role and importance of the parasympathetic nervous system. | `node_0008` |
| `obj_01_04` | 6. Use tools for better emotional regulation. | `node_0009` |

#### Skill 2 — Ethical Mindfulness

| Objective ID | Title | KB node |
|---|---|---|
| `obj_02_01` | 1. Define Ethical Mindfulness. | `node_0059` |
| `obj_02_02` | 2. Identify the three components of ethical mindfulness. | `node_0060` |
| `obj_02_03` | 3. Describe the significance of attention stability. | `node_0061` |
| `obj_02_04` | 4. Demonstrate how to practice mindfulness in daily activities. | `node_0062` |
| `obj_02_05` | 5. Analyze scenarios to determine how ethical mindfulness can influence outcomes. | `node_0063` |

#### Skill 3 — Emotional Awareness

| Objective ID | Title | KB node |
|---|---|---|
| `obj_03_01` | 1. Define emotional awareness. | `node_0097` |
| `obj_03_05` | 2. Explain the importance of emotional awareness. | `node_0098` |
| `obj_03_06` | 3. Identify afflictive and beneficial mental states. | `node_0099` |
| `obj_03_02` | 4. Identify the nature of your thoughts. | `node_0100` |
| `obj_03_03` | 5. Create space between speech and action to have greater mental balance and be less reactive. | `node_0101` |
| `obj_03_04` | 6. Apply relevant tools to practice calmness. | `node_0102` |

#### Skill 4 — Self Compassion

| Objective ID | Title | KB node |
|---|---|---|
| `obj_04_01` | 1. Define self compassion | `node_0151` |
| `obj_04_02` | 2. Identify the different thinking traps and how they impact well being | `node_0152` |
| `obj_04_05` | 3. Recognize the importance of self compassion | `node_0153` |
| `obj_04_06` | 4. Examine if you practice self compassion | `node_0154` |
| `obj_04_07` | 5. Describe the three steps to cultivate compassion | `node_0155` |
| `obj_04_03` | 6. Analyze how suffering and happiness depend on our thoughts and emotions rather than external situations | `node_0156` |
| `obj_04_04` | 7. Practice tools to develop greater self-compassion | `node_0157` |

The non-sequential IDs are not inferred from PDF numbering alone; they are the explicit compatibility mapping in `objective_mapper.py`.

### Mapping and reference checks

- Duplicate node IDs: **none**.
- Broken `parent_id` references: **none**.
- Broken `next_nodes` references: **none**.
- Broken `prerequisites` references: **none**.
- Content nodes without `skill_id`: **none**.
- Content nodes without `objective_id`: **8**:
  - `node_0011`, `node_0012`, `node_0013`, `node_0014`, `node_0015` — page 1, Skill 1 introduction/foundation.
  - `node_0065` — page 5, Skill 2 introduction.
  - `node_0104`, `node_0105` — page 8, Skill 3 introduction.
- Current `validation_status` counts: `approved` 197, `review` 8.
- Current `match_status` counts: `matched` 167, `not_applicable` 30, `review` 8.

### Manual-review nodes

| Node | Page | Title | Mapping status |
|---|---:|---|---|
| `node_0011` | 1 | The Road Trip Analogy | `skill_01`, no objective, `review` |
| `node_0012` | 1 | Mind-Body Maintenance | `skill_01`, no objective, `review` |
| `node_0013` | 1 | Foundational Skills for Well-Being | `skill_01`, no objective, `review` |
| `node_0014` | 1 | The Body and Mind Connection | `skill_01`, no objective, `review` |
| `node_0015` | 1 | Pacing & Future Integration | `skill_01`, no objective, `review` |
| `node_0065` | 5 | Introduction to Ethical Mindfulness scenario | `skill_02`, no objective, `review` |
| `node_0104` | 8 | The “I’m Fine” Habit | `skill_03`, no objective, `review` |
| `node_0105` | 8 | Consequences of Unmanaged Feelings | `skill_03`, no objective, `review` |

`validation_report.json` is inconsistent with the current JSON: it reports 205 approved nodes and 62 review mappings, while the current node fields contain 197 approved and 8 review nodes. It also reports no issues. This indicates that the report is stale or was generated by a different validation stage.

## 3. Curriculum and objective-ID consistency

### `knowledge_base.json`

Contains four PDF-derived skills and 24 unique objective IDs. The source objective numbering is retained in titles, while canonical IDs use the explicit mapping in `objective_mapper.py`.

### `progress_structure.json`

Contains four chapters and 24 objectives. It uses the same 24 IDs and the same source-number order as the knowledge base. Its summary reports:

- `chapter_count`: 4
- `objective_count`: 24
- `actionable_item_count`: 1

The structure’s objective `content` fields are composed from repeated objective/source text. This is retrieval-compatible but contains substantial duplication.

### `content_skills.json`

This is materially inconsistent with the PDF-derived curriculum:

- 8 skills instead of 4.
- 32 objectives instead of 24.
- Skills 5–8 are `Impartiality and Common Humanity`, `Forgiveness and Gratitude`, `Empathic Concern`, and `Compassion`, which are not represented as chapters in the current SEEK PDF knowledge base.
- Skills 1–4 use sequential IDs such as `obj_01_01` through `obj_01_04`, but their meanings do not consistently match the PDF objective numbering. For example, authored `obj_01_02` is “Define the resilient zone,” whereas the PDF-derived current `obj_01_02` is the PDF’s objective 3; authored `obj_01_04` is a tools objective, whereas the PDF-derived current `obj_01_04` is the PDF’s objective 6.

This is not merely formatting drift. The file is a separate, richer authored curriculum source and is still used by `/api/content/skills`; the code does not establish a migration proving that its 32-objective model is equivalent to the four-skill PDF model.

### `objective_mapper.py`

The explicit `_objective_id_for_number()` mapping is the clearest canonical compatibility table. It maps PDF objective numbers to the existing IDs, including non-sequential mappings:

- Skill 1: PDF 1→`obj_01_01`, 2→`obj_01_05`, 3→`obj_01_02`, 4→`obj_01_06`, 5→`obj_01_03`, 6→`obj_01_04`.
- Skill 2: PDF 1–5→`obj_02_01`–`obj_02_05`.
- Skill 3: PDF 1→`obj_03_01`, 2→`obj_03_05`, 3→`obj_03_06`, 4→`obj_03_02`, 5→`obj_03_03`, 6→`obj_03_04`.
- Skill 4: PDF 1→`obj_04_01`, 2→`obj_04_02`, 3→`obj_04_05`, 4→`obj_04_06`, 5→`obj_04_07`, 6→`obj_04_03`, 7→`obj_04_04`.

Content mapping is ancestry-based: a content node inherits the objective ID of its nearest learning-objective ancestor. Structural types are excluded from objective mapping.

### `progress_builder.py`

`build_progress_structure()` uses node `objective_id` values and chapter `skill_id` values, deduplicating repeated objective nodes. The separate `build_content_progress_structure()` still exists and builds a different authored-content model from `content_skills.json`; this creates an ongoing risk if the legacy function is called by another script.

### Existing compatibility logic

`server.py:resolve_legacy_progress_ids()` translates chapter node IDs and item/node IDs to progress-structure IDs. It does not provide a general objective-ID migration table. `complete_objective()` accepts only an objective ID present in the current progress structure. Existing records containing IDs from the alternate `content_skills.json` model therefore require explicit migration or a compatibility policy; none was found in the inspected code.

## 4. Progress-calculation logic

### Objective totals and progress formula

`ProgressService.calculate()` iterates `progress_structure.json["chapters"]`, so the current total is 24 objectives across four skills. For each objective:

1. Required item IDs are the intersection of `required_item_ids` and actual item IDs, or all item IDs if no required list exists.
2. `completion_state()` checks activity and quiz requirements.
3. An objective is complete if authored completion conditions pass **or** its objective ID appears in `completed_objective_ids`.
4. Otherwise, objective progress is:

`round(completed_required_item_count / total_required_item_count * 100, 1)`

If an objective has no progress items, its percentage is 0 unless it is explicitly marked complete.

Skill progress is the arithmetic mean of objective percentages. Overall progress is the arithmetic mean of skill percentages. The output also reports counts of fully completed skills/objectives, not a weighted item total.

### Activity and quiz rules

`completion_requirements()` treats `activity`/`practice` items as activity requirements and `quiz`/`assessment` items as quiz requirements. Invalid quiz passing scores default to **80**.

`completion_state()` requires:

- At least one required activity item completed.
- At least one required quiz passed at its configured threshold.

Both are required for automatic completion. The current progress structure reports only one actionable item and no quiz item types, so most or all of the 24 objectives do not have a complete activity-plus-quiz path in this structure.

### Routes and persistence

- `GET /api/progress/<skill_id>` loads one skill’s progress, calculates it, and selects the first uncompleted objective.
- `GET /api/progress` loads all progress documents and calculates the complete summary.
- `GET /api/progress/structure` returns the loaded progress structure.
- `POST /api/progress/item-complete` validates skill/item/objective associations, records item completion or quiz attempts, and adds an objective to `completed_objective_ids` only when `completion_state()` is complete.
- `POST /api/objectives/<objective_id>/complete` permits explicit completion only when the activity and quiz conditions pass, or when a chatbot quick check passes and activity completion is already true.

`LearningStore` stores per-student/per-skill documents with `completed_item_ids`, `completed_objective_ids`, `quiz_scores`, `quiz_attempts`, `learning_events`, and `next_recommended_learning_objective`. MongoDB is the primary store; an in-memory fallback is used if MongoDB is unavailable.

### Chatbot interactions and quick checks

`POST /api/learning-response` saves the interaction and records a chatbot learning event with status `introduced`. This does **not** complete an objective.

The objective-completion route can use a saved interaction’s `practice_check` and a boolean `quick_check_answer`, but it still requires `activity_completed`. A student merely asking a question or receiving a response is not treated as completion.

### Next-objective selection

`next_uncompleted_objective()` returns the first objective in the current progress chapter whose ID is not in the supplied completed set. The chatbot’s `_suggestion_target_for_progress()` walks the current and later chapters, skips completed IDs, and requires approved content nodes for the candidate objective before returning a suggestion.

### Compatibility risk

The current progress structure and current knowledge base agree on 24 IDs. However, `content_skills.json` remains a separate 32-objective curriculum and the code has no comprehensive migration for records written under that model. A deployed database containing authored-model IDs could therefore show objectives that are not found in the current progress structure.

## 5. Chatbot knowledge retrieval

`server.py` loads `knowledge_base.json`, `keyword_skill_map.json`, and `highlighted_keywords.json`. `/api/learning-response` calls `answer_student_question()` with the knowledge base, mapping data, saved progress, and the current progress structure.

The controller:

1. Detects terms and expands a small set of aliases.
2. Uses keyword-skill mappings when available, otherwise direct chapter scoring.
3. Falls back to semantic search over `type == "content"` nodes.
4. Ranks objective nodes within the selected skill.
5. Selects a relevant activity/practice/reflection.
6. Passes the selected context to the Gemini responder.

The current controller’s approved SEEK retrieval path uses the extracted knowledge base. However:

- `semantic_search()` only searches content nodes. Eight introductory content nodes have no objective ID, so content can be retrieved with a valid skill but no objective mapping.
- `/api/content/skills` serves `content_skills.json`, not the PDF-derived knowledge base. A frontend can therefore display a curriculum that differs from the chatbot/progress curriculum.
- The legacy `build_content_progress_structure()` remains available and is based on the conflicting authored source.
- The current progress structure has only one actionable item, so recommendation/activity availability is sparse even when textual content exists.

## 6. Existing student-progress data

The code is configured for MongoDB using `MONGODB_URI` and `MONGODB_DATABASE`, defaulting to `mongodb://localhost:27017` and `empathy_learning`. Relevant collections are:

- `student_progress`
- `learning_interactions`
- `controller_evaluations`

Progress documents use `student_id`, `skill_id`, `completed_item_ids`, `completed_objective_ids`, `learning_events`, `quiz_scores`, `quiz_attempts`, and `next_recommended_learning_objective`.

No database connection or read-only database query was executed during this audit. This avoids contact with potentially live student data and avoids exposing personal information. Database record consistency, actual stored objective IDs, and migration state therefore remain unverified.

## 7. Validation and testing

### Checks actually executed

The following read-only in-memory checks were executed against the current files:

- Parsed all four JSON files successfully.
- Counted 205 knowledge-base nodes.
- Confirmed 205 unique node IDs.
- Confirmed four chapter skills.
- Confirmed 24 unique knowledge-base objective IDs.
- Counted node types, validation statuses, and match statuses.
- Checked all `parent_id`, `next_nodes`, and `prerequisites` references; no broken references were found.
- Compared objective IDs and counts between the knowledge base, progress structure, and content-skills data.
- Enumerated content nodes missing objective IDs.

Results: the JSON/reference checks passed; the cross-file consistency comparison identified the conflicts documented above.

### Existing checks inspected but not executed

Existing tests and test modules were not executed because this audit was required to avoid commands that may write caches or other project files, and the environment previously denied test-process permission. No test suite pass is claimed.

The code contains validation/report generation in `app.py`, mapper tests, semantic-search tests, progress tests, and controller tests. The checked-in `validation_report.json` was not accepted as current evidence because its counts conflict with the current node fields.

## 8. Findings and risks

| Severity | File/function | Finding | Evidence | Potential impact | Recommended next action |
|---|---|---|---|---|---|
| High | `output/content_skills.json`; `server.py:load_content_skills`; `/api/content/skills` | Two competing curriculum models are active. | `content_skills.json` has 8 skills/32 objectives; KB/progress have 4 skills/24 objectives. | Frontend content, chatbot context, and progress can disagree; objective IDs may refer to different meanings. | Choose one canonical curriculum source and add an explicit migration/compatibility policy before changing records. |
| High | `output/progress_structure.json`; `ProgressService.completion_state` | Current progress structure has only one actionable item and no quiz item type. | Summary says `actionable_item_count: 1`, `item_types: ["practice"]`; completion requires activity plus quiz. | Most objectives cannot reach automatic completion through the current structure. | Reconcile the progress blueprint with the intended authored activity/quiz requirements and test every objective. |
| High | `output/validation_report.json` | Validation report is stale/inconsistent. | Report says 205 approved and 62 review mappings; current node fields say 197 approved and 8 review. | Review decisions may be based on inaccurate audit output. | Regenerate a report from the current JSON without overwriting it until reviewed. |
| Medium | `knowledge_base.json`; `semantic_search.py`/controller | Eight content nodes are retrievable by skill but have no objective ID. | Nodes `node_0011`–`node_0015`, `node_0065`, `node_0104`, `node_0105`. | Chatbot can provide source content without identifying a learning objective. | Manually classify these introductions or deliberately mark them as non-objective context and exclude them from objective progress claims. |
| Medium | `server.py:resolve_legacy_progress_ids` | Compatibility translates node/item IDs but has no general objective-ID migration. | Function handles chapter/item legacy IDs; no objective alias table found. | Existing records using the alternate authored curriculum may not resolve after a curriculum switch. | Inventory stored IDs read-only and define a versioned migration plan. |
| Medium | `progress_service.py`; `server.py:complete_objective` | Explicit completion can preserve a manually stored objective ID even when item-level evidence is incomplete only through the guarded quick-check/activity path; legacy records are trusted by `calculate()`. | `calculate()` treats IDs in `completed_objective_ids` as complete. | Historical or migrated IDs can show 100% without current item evidence. | Add provenance/version validation for legacy completion records before any migration. |
| Low | `progress_builder.py` | Two builders produce different structures. | `build_progress_structure()` uses KB nodes; `build_content_progress_structure()` uses `content_skills.json`. | Future scripts can regenerate the wrong model accidentally. | Deprecate or clearly isolate the legacy builder after migration decisions are made. |

## 9. Final summary

### Appears correct

- The current knowledge base parses as valid JSON.
- It has 205 unique node IDs.
- All four PDF skills and 24 unique objective IDs are represented.
- Parent, prerequisite, and next-node references resolve.
- Objective mapping is explicitly encoded in `objective_mapper.py`, including non-sequential compatibility IDs.
- Student questions and chatbot responses are not themselves treated as objective completion.
- Progress is persisted with separate item, objective, quiz, and event fields.

### Remains unverified

- Actual MongoDB progress records and their objective IDs.
- Whether frontend clients use `/api/content/skills` or `/api/progress/structure` as their curriculum source.
- Full runtime/API test behavior.
- Whether every source PDF content node should be assigned to an objective.
- Whether the alternate eight-skill authored curriculum is still intentionally supported.

### Confirmed defects

1. `content_skills.json` conflicts with the current four-skill/24-objective SEEK model.
2. `progress_structure.json` has only one actionable item while completion logic requires activity and quiz evidence.
3. `validation_report.json` does not describe the current node-level validation counts.
4. Eight content nodes lack objective mappings and can still be retrieved as skill-scoped content.
5. No general objective-ID migration was found for switching between the two curriculum models.

### Manual review items

`node_0011`, `node_0012`, `node_0013`, `node_0014`, `node_0015`, `node_0065`, `node_0104`, and `node_0105` require a curriculum-owner decision about whether they are shared introductory context or belong to specific objectives.

### Recommended order for future fixes

1. Decide and document the canonical curriculum source.
2. Inventory existing MongoDB objective IDs read-only.
3. Define a versioned migration/compatibility policy.
4. Reconcile progress items and completion requirements for all 24 objectives.
5. Resolve or explicitly classify the eight unmapped introductory nodes.
6. Regenerate validation output and add cross-source consistency tests.
7. Only then change frontend/API behavior or migrate stored records.

