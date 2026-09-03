# lead-verification

Rules-first classifier for inbound leads — call transcripts, SMS and web form submissions — running in production at roughly 1,000 leads a weekday.

Each lead needs three judgements: did a human answer, is this a genuine new-business enquiry, and is it spam.

## Why rules first

The obvious build is to post every transcript to a language model. This does the opposite: several hundred deterministic conditions handle everything decidable from the transcript itself, and the model is called only for what is left.

The reasoning is not that models are bad at this. It is that at a thousand leads a day, an LLM-per-lead architecture costs money on every classification, adds a network round trip to each one, can return different answers for identical input on different days, and leaves no way to explain a decision to the operations team beyond quoting the model back at them.

Rules are free, instant, deterministic and auditable. The model earns its place on the cases where a rule genuinely cannot be written.

Measured at **94% agreement with manual review**, across 1,000 randomly sampled production leads validated by hand. Not a holdout split on tuning data — a random sample of live traffic, checked by a person.

## What replaced what

This service took over from a 297-module visual automation with 33 separate model invocation points spread across its branches, nine levels of nesting, and a 9.3 MB export. It worked. It also could not be safely modified, because there was no way to see the whole thing at once or test a change without running it against production data.

The automation platform now does what it is good at — polling, dispatch, retry — in 40 modules. The classification logic lives here, where it can be read in one file and run against a labelled dataset.

| | Before | After |
|---|---|---|
| Modules in scenario | 297 | 40 |
| Model invocation points on canvas | 33 | 0 |
| Export size | 9.3 MB | 0.2 MB |

## Contents

- `lead_analyzer.py` — the classifier. Deterministic conditions, with a model fallback returning strict JSON against a fixed schema.
- `server.py` — Flask wrapper exposing single and batch endpoints with API key auth.

## Reliability

47,265 executions in the first 63 days of production, of which 47,264 succeeded. Median classification latency 2.4 seconds, 95th percentile 6.3 seconds.

## Known weaknesses

Published as it runs, not as it should be. In rough order of how much they bother me:

- **The API key is accepted from a query parameter as well as a header.** Query strings end up in access and proxy logs, which is a credential landing in files nobody classifies as sensitive. Should be header-only.
- **Key comparison is plain string equality**, not constant-time. `secrets.compare_digest` exists and `secrets` is already imported.
- **The rate limit delay ships at its testing value** of zero, with a comment noting the strict setting is thirteen.
- **`analyze_batch` accepts an unbounded array**, each element costing a potential model call.
- **Several hundred hand-tuned conditions are a maintenance liability.** Fast, free and explainable, but brittle against lead sources that did not exist when they were written. Tracking how often the model fallback fires would surface that drift early.
- **The 6% is uncharacterised.** Knowing the error rate is not the same as knowing whether the misses cluster in one source, or whether they skew towards disqualifying real leads rather than passing spam. Those have very different costs.
