# AI Tools Used

## Summary

I used **Claude Code (Claude Sonnet 5.5)** as a pair programmer and tutor, and **Gemini 3.5 Flash-Lite** as the LLM powering the student explanation judge.

Claude Code drafted most of the implementation and written documentation. I defined the project direction, made the key design decisions, ran and reviewed the notebooks, checked the results, managed the GitHub workflow, and used AI-assisted explanations to understand the implementation and prepare for the interview.

I have documented the AI assistance because being able to explain, evaluate, and modify the resulting system is as important as getting it to work.

## How I used Claude Code

| Area | AI assistance | My contribution |
|---|---|---|
| Planning and workflow | Helped break the task into 13 GitHub issues and suggest an implementation order | Created and managed the repository, issues, milestones, branches, commits, pull requests, and merges |
| Data science and ML | Drafted the cleaning, EDA, feature engineering, and model evaluation notebooks | Ran the notebooks, reviewed the cleaning decisions and outputs, checked the findings, and evaluated the modelling choices |
| LLM judge | Drafted `src/judge.py`, prompt versions v1 and v2, and evaluation notebooks | Configured the Gemini API, selected the model, ran the evaluations, reviewed failure cases, and assessed the limitations of the results |
| Research memo | Assisted with provider research, cost calculations, privacy comparisons, and drafting | Reviewed the recommendation, considered the trade-offs, and checked which claims needed qualification |
| Testing and review | Helped implement changes following an independent review of the repository | Evaluated the proposed changes and approved the ones I considered appropriate for the submission |
| Learning and debugging | Explained implementation details, evaluation metrics, and technical trade-offs | Used these explanations to review the work, investigate issues, and prepare to discuss and modify the code |

## Key technical decisions

I used AI assistance to explore implementation options, but reviewed the reasoning behind the main project decisions.

- **Dropout prediction:** Excluded `refund_requested` because it is strongly associated with the outcome and may only be known after a student has effectively dropped out. Using it could introduce target leakage.
- **Evaluation priorities:** Chose a recall-first approach because missing a student at risk of dropping out may mean missing an opportunity to intervene. I also checked the effect on precision and the number of students flagged.
- **Threshold selection:** Selected the 0.4 threshold as a recall-first policy choice, not because it demonstrated a confirmed improvement on the held-out test set. The test-set recall was 0.80 at both 0.4 and 0.5.
- **LLM prompt evaluation:** Developed prompt v2 after examining errors from v1. Since the same 60 labelled explanations informed prompt development, I treat the resulting 100% accuracy as a tuning-set result, not evidence of independent generalisation.
- **Small-model comparison:** Added a TF-IDF and Logistic Regression baseline. Used grouped cross-validation to reduce the optimistic bias caused by duplicate explanation texts, and distinguished agreement with the LLM from accuracy against human labels.
- **Scope management:** Considered moving categorical encoding into the model pipeline, but deferred that change because it would require regenerating the at-risk predictions and rerunning the downstream feedback workflow. I documented it as a potential improvement instead.

## How I checked the work

- Ran the notebooks in a local virtual environment and compared the outputs against the reported findings.
- Tested the repository from a fresh clone to check installation and reproducibility without requiring an LLM API key for the offline workflow.
- Reviewed the research memo's sources, cost assumptions, privacy claims, and stated uncertainties.
- Used mocked API responses for testing so the relevant error-handling and validation behaviour could be checked without relying on live API calls.

The Gemini API key is stored in a local `.env` file and is not committed to the repository. The LLM evaluation uses synthetic student data.

## Other AI tool

**ChatGPT** was used once as an independent reviewer of the completed repository. I reviewed its recommendations with Claude Code and decided which changes to adopt. I did not use ChatGPT to write the project code.

## LLM used in the product

**Gemini 3.5 Flash-Lite**, accessed through the Google AI Studio API free tier, powers `src/judge.py`. The research memo discusses the model's estimated cost, quality, privacy considerations, and language support.

## Limitations and next steps

The project is a prototype evaluated on a small synthetic dataset. Its results do not establish real-world dropout prediction performance or the LLM judge's accuracy on unseen explanations.

With more time, I would evaluate the frozen LLM prompt on newly labelled explanations, strengthen feedback-quality checks, validate the dropout model on a larger and more representative dataset, and improve the reproducibility of the prediction pipeline.

My goal in using AI tools was to accelerate implementation while reviewing the resulting work and developing an understanding of its design decisions, limitations, and possible improvements.