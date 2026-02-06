# EO Policy Tracker Evaluation Framework - Quick Intro Script

> Use this script when you need to quickly explain what the team is working on without doing a full demo.

---

## The 30-Second Version

"We stood up a **standalone evaluation framework** for the EO Policy Tracker. The goal is simple: **how do we systematically improve the system's accuracy?**

We met with Matt and Mike, and to avoid interfering with Tim and Varsha's production work, we decided we needed a separate environment to test and iterate on prompts.

What it does: we take the **golden dataset**—the 152 SME-validated policy-EO pairs from last May—run them through different prompt versions, measure accuracy, precision, recall, and F1 score, and then quickly iterate.

We just kicked this off in the last week, so we're still building, but the goal is **rapid, measurable improvement**."

---

## The 60-Second Version

"So here's the situation: we have this EO Policy Tracker that compares policies against Executive Orders. It works, but we wanted a way to **systematically improve it** without breaking what's already in production.

We met with Matt and Mike, and the consensus was: **build a standalone evaluation framework**. That way we're not touching Tim and Varsha's code, but we can still experiment.

What we built:
- A **modular pipeline** that runs policies through three phases: classification, reasoning, and justification comparison
- **Prompt versioning**, so we can test V1 vs V2 vs V3 and see which performs better
- **Metrics tracking**: accuracy, precision, recall, F1—all saved and comparable across runs

Matt specifically asked us to look at the **quality of the justifications**, not just whether the final flag is correct. So we added a Phase 3 that compares the AI's reasoning to what the SMEs wrote.

The workflow is: run an evaluation, find the failures, generate an improved prompt, test it, save it, and run again. We're trying to make that iteration loop as fast as possible.

We just started this, so we'll see where it goes."

---

## Key Talking Points

- **Why standalone?** → Don't interfere with production work
- **Why now?** → Need measurable improvement, not guessing
- **What's unique?** → Phase 3 compares AI reasoning to SME reasoning
- **Where are we?** → Just kicked off this past week
- **What's next?** → Keep iterating, track metrics over time
