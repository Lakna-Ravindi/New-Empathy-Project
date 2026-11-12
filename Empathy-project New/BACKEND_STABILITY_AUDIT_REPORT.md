# Empathy Learning System - Backend Stability Audit

**Audit type:** Read-only analysis and testing  
**Audit date:** 2026-11-06  
**Scope:** `backend/`, generated curriculum files in `output/`, and read-only
inspection of the configured MongoDB database  
**Changes made:** No source code, configuration, curriculum JSON, or database
records were modified. This report is the only file created.

## 1. Overall assessment

**Assessment: NOT READY for Agentic AI integration.**

The backend imports successfully and MongoDB is reachable, but the system is
not stable enough to use as an agentic foundation. The most important blockers
are:

1. Populated API/database/JWT credentials are stored in plaintext environment
   files. These credentials must be treated as compromised and rotated.
2. The canonical progress structure has 24 objectives but only one actionable
   item. Most objective completion checks therefore cannot represent the
   authored curriculum.
3. Existing MongoDB progress contains legacy node IDs, skill IDs outside the
   current structure, and item/objective IDs not present in the current
   blueprint.
4. The full importable test run produced **106 passed, 9 failed, 12 subtests
   passed**. Full collection also fails before running because two test modules
   use non-package imports.
5. The implementation uses an 80% default quiz threshold and “any activity”
   completion. No implementation of the requested 50% activity and 90% quiz
   eligibility rules was found.
6. The unauthenticated reviewer evaluation endpoints can write evaluation data
   and expose summary data.

No component is marked fully working below unless the relevant behavior was
verified by code inspection and/or tests.

## 2. Validation performed

### Environment and startup

- `python -m compileall -q .`: passed.
- `import server`: passed.
- Runtime initialization reported:
  - 8 content skills loaded.
  - 205 knowledge-base nodes loaded.
  - authentication MongoDB store available.
  - learning MongoDB store available.
- Installed versions were checked against
  [`requirements.txt`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/requirements.txt>).
  Pinned packages matched the installed environment. `google-genai` and
  `pymongo` are unpinned, so reproducibility is not guaranteed.
- `backend/.env` and the project-root `.env` contain populated values for
  Gemini, MongoDB, JWT, and Flask/session secrets. Values are intentionally not
  reproduced in this report.

### Tests

Command:

```text
python -m pytest -q --ignore=classifier/test_classifier.py --ignore=knowledge/test_node.py
```

Result: **106 passed, 9 failed, 12 subtests passed** in 19.78 seconds.

Full collection additionally fails during collection:

- [`classifier/test_classifier.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/classifier/test_classifier.py>)
  imports `rule_classifier` as a top-level module.
- [`knowledge/test_node.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/knowledge/test_node.py>)
  imports `node_builder` as a top-level module.

These are packaging/test-discovery defects, not evidence that the underlying
components pass.

Failed tests:

- [`learning/test_gemini_responder.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/learning/test_gemini_responder.py>):
  missing activity fallback.
- [`learning/test_pedagogical_controller.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/learning/test_pedagogical_controller.py>):
  actionable-question handling, valid reflection selection, unavailable
  activity handling, persisted next-objective selection, primary activity
  selection, retrieval without keyword records, and completed-objective
  skipping.
- [`learning/test_semantic_search.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/learning/test_semantic_search.py>):
  emotional-awareness objective selection.

## 3. Priorities

| Priority | Finding |
|---|---|
| Critical | Plaintext live credentials in `.env` files; rotate all affected secrets and invalidate tokens. |
| Critical | Unauthenticated evaluation write/read endpoints in [`server.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/server.py>). |
| Critical | Progress data and current curriculum use incompatible ID namespaces. |
| High | Progress structure has 24 objectives but only one actionable item. |
| High | Learning-controller and Gemini-response tests fail in core recommendation paths. |
| High | Requested 50% activity / 90% quiz rules are absent; default quiz rule is 80%. |
| High | MongoDB fallback can silently move a running instance to process-local storage. |
| Medium | Full test collection is broken by import/package layout. |
| Medium | API request validation is incomplete for learning-response and progress routes. |
| Medium | Screening thresholds are explicitly marked as needing clinical-protocol alignment. |
| Low | Duplicate/legacy entry points (`app.py`, `server_backup.py`, `app/`) increase configuration and maintenance ambiguity. |

## 4. Environment and infrastructure findings

### ENV-001 - exposed credentials (Critical)

- **Location:** [`backend/.env`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/.env>) and project-root `.env`.
- **Evidence:** populated `GEMINI_API_KEY`, MongoDB URI, JWT secret, and Flask
  secret are present. `server.py`, `auth.py`, and
  [`gemini_responder.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/learning/gemini_responder.py>)
  load these values.
- **Impact:** anyone with repository/workspace access can impersonate users,
  call Gemini, or access the database. Existing JWTs may remain forgeable until
  rotation.
- **Recommendation:** rotate/revoke all exposed values immediately, remove
  secrets from tracked/local project files, use a secret manager, and fail
  closed when required secrets are absent.

### ENV-002 - dependency reproducibility (Medium)

- **Location:** [`requirements.txt`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/requirements.txt>).
- **Evidence:** `google-genai` and `pymongo` have no version constraints.
- **Impact:** a future install can change SDK behavior or break response parsing.
- **Recommendation:** pin tested versions and generate a lock/constraints file.

### ENV-003 - multiple configuration loaders (Medium)

- **Locations:** [`server.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/server.py>),
  [`auth.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/auth.py>),
  [`app/core/security.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/app/core/security.py>).
- **Evidence:** server-side modules load different `.env` paths; the legacy
  FastAPI security module uses a different token configuration from Flask
  `auth.py`.
- **Impact:** behavior depends on working directory/import path and can produce
  tokens that another component cannot validate.
- **Recommendation:** choose one supported entry point and one explicit,
  absolute configuration loader.

### ENV-004 - silent local fallback (High)

- **Location:** [`learning/interaction_store.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/learning/interaction_store.py>),
  `LearningStore.__init__`.
- **Evidence:** a broad `except Exception` marks MongoDB unavailable and
  switches to process-local dictionaries.
- **Impact:** a production outage can appear successful while interactions and
  progress are lost on restart or split across workers.
- **Recommendation:** use explicit connection errors, a health/readiness
  failure for production, and only enable local fallback under an explicit
  development setting.

### ENV-005 - debug startup (High)

- **Location:** [`server.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/server.py>),
  `app.run` near line 1606.
- **Evidence:** `debug=True` is hard-coded.
- **Impact:** unsafe deployment defaults and possible sensitive traceback
  exposure.
- **Recommendation:** use an environment-controlled setting and a production
  WSGI server.

## 5. Knowledge-base and mapping findings

### KB-001 - current JSON has duplicate-free nodes but conflicting exports (High)

- **Evidence:** `knowledge_base.json` has 205 unique node IDs and no broken
  parent references. However:
  - `content_skills.json` contains 8 skills and 32 objectives.
  - `progress_structure.json` contains only skills 1-4 and 24 objectives.
  - skills 5-8 and their 16 objectives are missing from the progress structure.
  - the progress structure has objectives absent from `content_skills.json`
    (including `obj_01_05`, `obj_01_06`, `obj_02_05`, and several skill 3/4
    objectives).
- **Impact:** content APIs, chatbot recommendations, and progress APIs do not
  describe one curriculum.
- **Recommendation:** establish one approved SEEK export and validate every
  generated artifact against it before publishing.

### KB-002 - actionable-item loss (Critical)

- **Location:** [`output/progress_structure.json`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/output/progress_structure.json>).
- **Evidence:** 24 objectives are present, but only `obj_04_04` has one item;
  all other objectives have zero items and zero required item IDs.
- **Impact:** completion, activity eligibility, quiz eligibility, and progress
  percentages cannot represent nearly the entire learning journey.
- **Recommendation:** stop relying on this artifact until item-to-objective
  generation is corrected and validated with counts per objective.

### KB-003 - keyword map uses node IDs as skill IDs (High)

- **Location:** [`output/keyword_skill_map.json`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/output/keyword_skill_map.json>).
- **Evidence:** mapped records use `node_0001`, `node_0051`, `node_0085`, and
  `node_0130` in the `skill_id` field, while the current skill IDs are
  `skill_01` through `skill_08`. Eleven of 32 records are not mapped.
- **Impact:** skill routing can select a node as a skill, producing
  recommendations that cannot be resolved by progress APIs.
- **Recommendation:** regenerate mappings using canonical `skill_id` and
  `objective_id` fields; reject records whose references are not in the
  approved manifest.

### KB-004 - stale validation report (Medium)

- **Location:** [`output/validation_report.json`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/output/validation_report.json>).
- **Evidence:** current node fields contain 197 approved and 8 review nodes,
  while the stored report says 205 approved and reports no issues.
- **Impact:** operators can rely on an inaccurate “valid” report.
- **Recommendation:** version validation output with the source hash and
  regenerate it as part of the build.

### KB-005 - source mapping still requires SEEK approval (Medium)

- **Locations:** [`objective_mapper.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/knowledge/objective_mapper.py>)
  and [`keyword_skill_mapper.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/knowledge/keyword_skill_mapper.py>).
- **Evidence:** mappings are heuristic/generated and the data contains
  review-status nodes. The audit can identify broken references, but it does
  not invent or approve new SEEK mappings.
- **Impact:** agentic systems could amplify an incorrect curriculum relation.
- **Recommendation:** obtain an approved mapping manifest and add automated
  rejection tests for unapproved relationships.

## 6. Learning logic findings

### LEARN-001 - core selection behavior is regression-prone (High)

- **Locations:** [`pedagogical_controller.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/learning/pedagogical_controller.py>),
  [`semantic_search.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/learning/semantic_search.py>).
- **Evidence:** 8 pedagogical-controller tests and 1 semantic-search test fail.
  Failures include selecting `skill_04` where a node ID was expected,
  selecting the wrong next objective, and index/`None` errors when activity
  data is absent.
- **Impact:** students can receive an unrelated skill/objective or no valid
  activity; progress can advance against a different namespace.
- **Recommendation:** define canonical selection output types, add invariant
  checks before response generation, and resolve the failing tests before
  agentic orchestration.

### LEARN-002 - Gemini fallback tests fail (High)

- **Location:** [`gemini_responder.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/learning/gemini_responder.py>),
  `generate_educational_response`.
- **Evidence:** the missing-activity fallback test fails, and fallback paths
  return different shapes depending on controller status. Gemini retries and
  fallback are present, but the contract is not consistently verified.
- **Impact:** frontend consumers and future agents may receive a string where a
  structured response is expected.
- **Recommendation:** specify one response schema for success, unavailable
  content, provider failure, and malformed provider output; test every path.

### LEARN-003 - duplicate decision surfaces (Medium)

- **Locations:** [`server.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/server.py>),
  [`pedagogical_controller.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/learning/pedagogical_controller.py>),
  [`gemini_responder.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/learning/gemini_responder.py>).
- **Evidence:** the controller selects learning context and activities, while
  the responder validates/rewrites activity metadata and generates suggested
  questions. Server-side ID attachment can also synthesize a recommended item.
- **Impact:** conflicting authorities make it difficult to prove that an agent
  cannot select an unapproved objective or activity.
- **Recommendation:** keep selection in one deterministic controller and make
  the provider a schema-constrained wording component only.

## 7. Progress tracking findings

### PROG-001 - eligibility rules do not match the requested rules (Critical)

- **Location:** [`progress_service.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/learning/progress_service.py>),
  `completion_requirements` and `completion_state`.
- **Evidence:** activity completion is `any` activity item; quiz completion
  uses a per-item `passing_score` or `DEFAULT_QUIZ_PASSING_SCORE = 80`.
  Repository search found no 50% activity threshold and no 90% quiz threshold.
- **Impact:** objectives may complete with one activity regardless of the
  requested 50% rule, and a score from 80-89 can pass when 90% is required.
- **Recommendation:** confirm the intended rule definition, implement it once
  in the progress service, and cover boundary cases 49/50/89/90.

### PROG-002 - Mongo completion path has an undefined variable (High)

- **Location:** [`interaction_store.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/learning/interaction_store.py>),
  `_objective_requirements_met`.
- **Evidence:** the generator expression compares against `passing_score`,
  but that name is not defined in the function; the iterator variable is
  named `passing_score` only inside a different expression scope.
- **Impact:** qualifying Mongo-backed completion can raise `NameError` rather
  than persist completion. Existing tests do not exercise this exact path
  with a live Mongo store.
- **Recommendation:** fix only after approval, then add a Mongo-backed
  regression test for passing and failing quiz scores.

### PROG-003 - persisted records contain incompatible IDs (Critical)

- **Evidence:** read-only MongoDB inspection found 38 `student_progress`
  documents with 38 unique student/skill pairs, but records use legacy skills
  such as `node_0001` and `node_0172`, skills 5-8 not in the current progress
  structure, and completed item/objective IDs absent from the current
  structure.
- **Impact:** calculated progress and next-objective recommendations cannot be
  trusted across existing students.
- **Recommendation:** define a versioned migration/compatibility policy,
  produce a dry-run reconciliation report, and obtain approval before any
  record migration. No records were changed in this audit.

### PROG-004 - inconsistent document shapes (High)

- **Evidence:** among 38 records, 9 omit `completed_item_ids`, 6 omit
  `completed_objective_ids`, 9 omit `learning_events`, 13 omit `quiz_scores`,
  20 omit `quiz_attempts`, and 6 omit the next-recommendation field.
- **Impact:** readers repair fields in memory, but different routes can observe
  different schemas and updates can overwrite legacy state.
- **Recommendation:** enforce a schema/version at write time and test legacy
  document normalization explicitly.

### PROG-005 - likely duplicate event semantics (Medium)

- **Location:** [`interaction_store.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/learning/interaction_store.py>),
  `complete_item`, `record_learning_event`, and `_merge_learning_event`.
- **Evidence:** `complete_item` writes progress and then calls
  `record_learning_event`, which reads and updates the same progress document
  again. The event is de-duplicated by item ID, but the two writes are not
  atomic.
- **Impact:** concurrent requests can lose fields or produce divergent event
  status and completion state.
- **Recommendation:** use one atomic update or a transaction and add
  concurrent/retry tests.

### PROG-006 - next-objective data is not consistently derived (High)

- **Locations:** [`server.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/server.py>),
  progress routes; [`progress_service.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/learning/progress_service.py>).
- **Evidence:** some routes calculate a next objective from verified current
  completion, while `complete_item` and `complete_objective` also persist a
  caller/controller-supplied next object.
- **Impact:** stale or incompatible recommendations can remain in MongoDB.
- **Recommendation:** derive the next objective centrally from canonical
  progress and validate that it belongs to the same skill/objective manifest.

## 8. API and security findings

### API-001 - evaluation endpoints are unauthenticated (Critical)

- **Location:** [`server.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/server.py>),
  `evaluate_interaction` near line 1555 and `evaluation_summary` near line
  1592.
- **Evidence:** these routes have no `@require_auth` or reviewer/admin role
  decorator, unlike the student routes.
- **Impact:** an unauthenticated caller can submit reviewer evaluations and may
  read evaluation summaries, corrupting QA data and exposing metadata.
- **Recommendation:** require authenticated reviewer/admin authorization,
  validate reviewer identity from the token, and enforce interaction ownership
  or reviewer scope.

### API-002 - learning-response input validation is incomplete (Medium)

- **Location:** [`server.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/server.py>),
  `learning_response`.
- **Evidence:** `question` is stripped and checked for emptiness but has no
  maximum length, unlike general empathy chat. The route accepts the JSON
  object fallback and forwards arbitrary-size text to retrieval/Gemini.
- **Impact:** resource exhaustion, excessive provider cost, and inconsistent
  error behavior.
- **Recommendation:** enforce a documented message size, content type,
  request limit, and provider timeout.

### API-003 - error handling is inconsistent (Medium)

- **Evidence:** [`server.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/server.py>)
  uses a broad `except Exception` around progress structure loading, while
  other routes catch narrower database errors. `interaction_store.py` also
  catches all exceptions during initialization.
- **Impact:** programming/configuration defects can be converted into
  misleading success-shaped fallback behavior.
- **Recommendation:** catch expected exception classes, log correlation IDs,
  and return explicit readiness/service errors.

### API-004 - screening safety policy is not independently verified (High)

- **Location:** [`registration_assessment.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/registration_assessment.py>),
  `score_screening`; [`server.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/server.py>),
  registration routes.
- **Evidence:** red-tier users are blocked from registration and screening
  records are stored, but the code contains a TODO stating thresholds must be
  aligned with the clinical protocol/ethics approval. No test evidence verifies
  counselor referral delivery or follow-up state transitions.
- **Impact:** a safety classification can be clinically unsuitable or provide
  no reliable referral workflow.
- **Recommendation:** obtain approved protocol thresholds, test every boundary
  and acute-risk combination, and verify an operational referral path.

### API-005 - logout is presentation-only (Low)

- **Location:** [`server.py`](<C:/Users/User/Desktop/New folder/New-Empathy-Project/Empathy-project New/backend/server.py>),
  `logout`.
- **Evidence:** the endpoint returns success but does not revoke tokens; JWTs
  remain valid until expiry.
- **Impact:** stolen tokens remain usable after logout.
- **Recommendation:** use short-lived access tokens with refresh-token
  rotation/revocation or a server-side denylist.

## 9. MongoDB audit

The configured MongoDB endpoint responded to a read-only ping. Collections
observed included `users`, `student_progress`, `learning_interactions`,
`screening_results`, `pre_skill_assessments`, `controller_evaluations`,
`safety_red_flags`, `student_safety_states`, `content_skills`,
`progress_structure`, and `learning_content`.

Observed counts (read-only):

- `student_progress`: 38
- `learning_interactions`: 381
- `users`: 34
- `screening_results`: 9
- `pre_skill_assessments`: 9
- `controller_evaluations`: 1

The audit did not print student message bodies, credentials, password hashes,
or identifying record contents. Existing records were not modified.

The collection set itself shows multiple generations of content/progress
storage (`students` and `users`, node-based and skill-based progress). This
should be treated as a data-model migration concern before agentic writes are
allowed.

## 10. Missing test coverage

The repository has useful unit tests for parts of controller, responder, and
progress behavior, but the following coverage is missing or insufficient:

- automated integrity validation for all JSON IDs, references, objective
  mappings, required items, and cross-file manifests;
- approved-SEEK mapping fixtures and rejection of unapproved mappings;
- boundary tests for the required 50% activity and 90% quiz rules;
- live-Mongo or Testcontainers integration tests for atomic progress updates,
  legacy document normalization, and the undefined-variable path;
- API authentication/authorization matrix for every route, especially
  evaluation endpoints and cross-student access;
- request-size, malformed JSON, provider timeout, rate-limit, and malformed
  Gemini response tests;
- end-to-end student journey from registration and screening through content,
  activity, quiz, objective completion, next objective, and history;
- safety referral delivery and red/amber/green boundary tests;
- concurrency/idempotency tests for repeated completion and retry requests;
- a clean test-package configuration so full collection runs without import
  errors.

## 11. Recommended implementation order (after approval)

1. Rotate all exposed credentials and remove secrets from project files.
2. Restrict evaluation routes and verify authentication/role/ownership tests.
3. Freeze agentic integration and establish one canonical skill/objective/item
   manifest approved against SEEK.
4. Rebuild and validate the progress structure; do not migrate MongoDB records
   until a dry-run reconciliation is reviewed.
5. Resolve the 50%/90% eligibility specification and implement boundary tests.
6. Correct progress persistence atomicity and add live-database integration
   coverage.
7. Fix controller/responder regressions and enforce one response schema.
8. Add request limits, explicit provider/database failure behavior, and
   production-safe startup configuration.
9. Repair test package imports and require a clean full-suite run.
10. Only then introduce agentic capabilities behind feature flags, with
    canonical objective authorization and audit logging.

## 12. Final readiness decision

**Agentic AI introduction: NO-GO at this time.**

The backend has a reachable runtime and some passing unit coverage, but its
curriculum identity model, progress records, eligibility rules, security
boundaries, and regression suite are not sufficiently reliable. Agentic
capabilities would increase the impact of wrong-objective selection, unsafe
data writes, and non-deterministic recommendations. Reassess readiness only
after the critical/high findings are resolved and the full test and
data-reconciliation gates pass.
