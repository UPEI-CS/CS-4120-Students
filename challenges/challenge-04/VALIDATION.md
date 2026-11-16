# Verification record · 16 November 2026

## Passed

- Actual official UCI archive download, filtering, temporal snapshots, all six candidates, selected-model SHAP and five fixed-forest seeds completed on the Mac in approximately 33 seconds. This is not a classroom/Colab timing guarantee.
- Exact SHAP checked against a known additive model and a constant baseline; additivity verified for every recorded explanation.
- Temporal boundary and future-feature invariance checks; repeated-run determinism; notebook syntax/schema and embedded-source parity; no test-period scoring code.
- Actual ipywidgets Python callbacks: failed preparation retry, failed computation retry, completion, repeated run clicks without retraining, repeated exports of unchanged evidence.
- JavaScript DOM-stub checks: real-run loading, all customer/seed views, repeated tab changes, reload, malformed schema/additivity rejection, count-markup injection and missing-field rejection.
- Interrupted download leaves the final cache untouched; invalid uploaded ZIP receives explicit replacement guidance.
- Both real-data three-panel plot figures rendered and visually inspected. Word answer-template pages rendered and visually inspected. Relative README links checked.

The recorded reference viewer/JSON contains actual validation-only evidence, with archive hash, seeds, feature/background/case definitions, software versions and a run ID. It is labelled as a reference run, not a student's own run. No final-test scores or customer identifiers are included.

## Not verified

- Fresh Google Colab execution, browser frontend widget rendering and download behaviour.
- Full browser visual/mobile rendering of the HTML viewer. The browser tool rejected local file URLs; the restriction was not bypassed. JavaScript DOM-stub tests are not browser rendering tests.
- Classroom timing and final alignment with the instructor's official grading criteria.

These files are ready for instructor review, not a claim of a fully rehearsed classroom release. Nothing has been pushed or deployed by this work. The older private draft was left untouched.
