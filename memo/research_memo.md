# Research memo: Student Success Assistant

Simran Duggal · 24 Oct 2026 · prices and policies checked 22–24 Oct 2026 (links at the end)

## Question A: Which LLM for Part 2, at 50,000 students a month?

**Recommendation: Gemini 3.5 Flash-Lite on the paid tier** (through Vertex AI in Mumbai if the contract allows), with Claude Haiku 5.5 as the cheaper challenger to test before launch. It is the only model I tested end to end, it costs about $38 a month, and the paid terms do not use student text to improve Google products. The free tier is not acceptable for student data: Google says free-tier content can be used to improve its products and read by human reviewers.

**How I costed it.** My judge (prompt v2) used about 590 input and 70 output tokens per call (5 live calls); I assume 600 and 80 for every provider, with ±20% for different tokenizers. Volume: 50,000 students × 2 explanations a month = 100,000 calls, which is generous (the course data has 1.45 explanations per student over four months). That is 60M input and 8M output tokens a month.

| | Gemini 3.5 Flash-Lite | Claude Haiku 5.5 | GPT-5.4 mini |
|---|---|---|---|
| Price per 1M tokens (in / out) | $0.30 / $2.50 | $0.10 / $0.50 | $0.75 / $4.50 |
| **Monthly cost**, 100k calls | 60×0.30 + 8×2.50 = **$38** | 60×0.10 + 8×0.50 = **$10** | 60×0.75 + 8×4.50 = **$81** |
| With Batch API (50% off, all three) | $19 | $5 | $41 |
| At 5 explanations per student | $95 | $25 | $203 |
| Speed | 1.0–1.7 s per call (measured); a monthly job needs no real-time speed: 100k calls at 10 in parallel is about 4 hours, or use batch | not measured | not measured |
| Quality on this task | Tested on 60 labelled rows: 73% with a simple prompt, 100% after tuning on the same rows (optimistic); all replies valid JSON; resisted 3 prompt-injection texts | not tested | not tested |
| Privacy | Paid: not used to improve products; logs kept for a limited period and "may be stored transiently or cached in any country"; Mumbai region listed, but Google says endpoints do not guarantee in-region processing | Commercial terms: no training on customer content; first-party API has no India region (third-party guide) | Not used for training by default; logs up to 30 days; India residency is storage only, with processing outside India |
| Hindi, Tamil, Kannada | No published numbers found. One Hinglish explanation judged correctly (a single example) | Hindi 92.4% of English score (Haiku 4.5, translated MMLU); Tamil and Kannada not published | Not found |

Cost is not the deciding factor: the three options differ by about $70 a month at this volume, less than an hour of engineer time. The decision rests on quality, privacy and language, which is why I would run the same 60-row test on Haiku 5.5 and GPT-5.4 mini before committing. Judging only the at-risk students (36% in Part 1) would cut every cost by about two thirds.

**Two risks.**
1. **Model churn.** During this task `gemini-2.5-flash` and `gemini-2.5-flash-lite` returned "no longer available to new users", `gemini-3.8-flash` timed out even on a one-word request, and its price doubles on 1 Jan 2027. Pin the model version, keep the labelled rows as a regression test, and rerun them on every change.
2. **Unverified language quality and data residency.** I found no Tamil or Kannada benchmark for any of the three models, and no written commitment that data stays in India (a third-party guide says India's DPDP obligations mainly start in May 2027). Before launch: test at least 50 labelled explanations per language, and get residency confirmed in writing.

**What I could not confirm.** Tamil and Kannada quality for any model; Haiku 5.5 and GPT-5.4 mini on this task and their speed (no API keys); Anthropic's retention period and India region from its own documents (third-party source only); what OpenAI's "luna" models are (listed, not used); prices are list prices and exclude tax and volume discounts.

## Question B: When should a company use a small ML model instead of an LLM?

**My view.** Use a small model when the task is narrow and fixed (the same inputs, a fixed set of answers), labelled data exists, volume is high, and you need cheap, fast, explainable decisions. Use an LLM when the input is open-ended language, there is little or no labelled data, the answer must be written text, or the task changes often. Two real examples where small models win: **Stripe Radar** scores every card payment from behavioural signals, with fraud labels taken from banks, across 100,000+ businesses; and **Gmail's spam filter**, where Google reported TensorFlow models blocking about 100 million extra spam messages a day on top of the over 99.9% already stopped. A 2026 preprint on four text-classification datasets found fine-tuned BERT-family models competitive with or better than prompted LLMs at one to two orders of magnitude lower cost and latency, and recommends LLMs as complements.

| | Small ML model | LLM |
|---|---|---|
| Cost | Close to zero per prediction | Per-token fees (here about $0.0004 a call) |
| Speed | Microseconds to milliseconds (mine: 0.05 ms) | Seconds (mine: 1.3 s) |
| Accuracy | Strong on narrow, stable tasks; weak on new wording | Handles new wording, languages and context |
| Explainability | Feature importance, coefficients | Explanations are text it generates, not its real reasoning |
| Data needed | Hundreds to thousands of labelled rows | None to a few examples |
| Maintenance | Retrain when data drifts | Re-test when the vendor changes or retires the model |

**Applied to this task.**
- **Part 1, dropout: a small model.** The data is 500 rows of numbers. The random forest reaches F1 0.71 (baseline 0.00), trains in seconds, costs nothing, and its feature importance shows what drives dropout (activity, not scores). An LLM would add cost and lose explainability with nothing to gain.
- **Part 2, judging explanations: an LLM.** The bonus TF-IDF + logistic regression scored 0.72 in grouped cross-validation, close to my first LLM prompt (0.73), and agreed with the tuned LLM on 90% of the other explanations (86% on new wording). It costs nothing and takes 0.05 ms. But it cannot write the feedback, which is the real product, and it labelled 14 of the 46 answers the LLM calls wrong as "partial", because it matches words and not meaning. With only 60 rows (52 distinct texts) the classifier score swung from 0.58 to 0.80 across reshuffles, so I would not trust either number much.
- **Best design: both.** The ML model decides who needs help; the LLM writes the feedback for them.

## Sources
- Gemini pricing (updated 22 Oct 2026): https://ai.google.dev/gemini-api/docs/pricing · terms: https://ai.google.dev/gemini-api/terms · Vertex locations: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
- Claude pricing: https://platform.claude.com/docs/en/about-claude/pricing · multilingual: https://platform.claude.com/docs/en/build-with-claude/multilingual-support · commercial terms: https://www.anthropic.com/legal/commercial-terms
- OpenAI pricing: https://developers.openai.com/api/docs/pricing · data controls: https://developers.openai.com/api/docs/guides/your-data
- India data residency, third-party guide: https://www.techplained.com/llm-api-data-residency-india
- Stripe Radar primer: https://stripe.com/in/blog/a-primer-on-machine-learning-for-fraud-detection · Gmail spam: https://9to5google.com/2019/02/06/gmail-tensorflow-spam-protections/ · Cost-aware model selection preprint: https://arxiv.org/abs/2602.06370
- My own measurements: notebooks 05–07 and `memo/research_notes.md`.
