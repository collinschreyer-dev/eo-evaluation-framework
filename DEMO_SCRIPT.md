# EO Evaluation Framework - Demo Script

## Opening: The "Why"

"Okay, let me walk you through what we've been building. This really stems from the question: **how do we improve this system?** We need to break it apart and make it as modular as possible.

We met with Matt and Mike to discuss this problem, and the conclusion was clear—to avoid interfering with the work that Tim and Varsha have already done, we needed a **standalone evaluation framework**. That's where this came from.

We just kicked this off yesterday, so let me show you what we're setting up here."

---

## Demo Flow

### 1. Getting Started

"Everything you see here will be packaged up with scripts to get it running on any machine. When you open the EO Evaluation Framework, you can immediately:

- **Select your model** from the ones available in US AI
- **Provide your US AI API key**
- **Configure the phases**

The original system was a two-phase approach. We added **Phase 3**, which compares the AI's justification with what the subject matter experts wrote—since we're using the **Golden Dataset**, which is really the best evaluation data we have, going back to last May."

### 2. Running an Evaluation

"If you just want a quick test, you can run **10 samples**. Or, if you're curious how changes will impact accuracy across the board, run a **full 152-record sample**.

Click run, and it processes through all three phases. Then you can check results in **Analyze & Debug** or **Results History**—everything is saved."

### 3. Data Persistence

"For persistence, we have:
- **SQLite database**
- **JSON exports**
- **CSV exports**

We'll make it fully configurable—whatever format you need for deeper analysis."

---

## Analyzing Results

"Now here's the big thing—let's analyze the results.

I just had a run, and you can see all the records. **Green** means it matched ground truth. **Red** means it didn't.

Let me grab one that's red so we can dig in..."

### Record Details View

"Now you see **two columns**:
- **Ground Truth**: What the policy expert or SME said
- **AI Response**: What the model predicted

This one says 'Not Affected'—and that's **wrong**."

### Phase 3 Justification Comparison

"Matt specifically requested we look at the justification quality, so we added this. It's another LLM-generated analysis that shows:
- **Similarity Score** (out of 100)
- **Core rationale alignment**
- **Specific differences identified**

Based on this, we can **generate a suggested prompt**. The model failed because of X, so here's a refined prompt that addresses that weakness."

---

## Iterative Prompt Refinement

"Here's where it gets powerful. You can:

1. **Copy the suggested prompt** to the Quick Prompt Tester
2. **Test it** with Gemini or Sonnet
   - In my testing, Gemini tends to be better at this type of evaluation than Anthropic, for whatever reason
3. **Run another similarity analysis** if you want to go even deeper

The workflow is really thorough—you can go as deep as you need.

Then you just hit **'Save as New Prompt'**, and now you have a V2."

---

## Full Iteration Cycle

"Now watch—go back to **Run Evaluation**, and you'll see **Phase 2 V2** in the dropdown.

You can now run a full evaluation with your new prompt, see the new accuracy, and continue iterating.

We just kicked this off in the last week here, so we'll see where it goes. But the goal is to make this **rapid iteration** process as smooth as possible."

---

## Closing

"Questions?"
