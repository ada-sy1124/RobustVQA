verification_prompt_template="""You are required to act as a logic checker. Your task is to analyze the provided Reasoning and determine the correct option for the Question based **STRICTLY AND ONLY** on the information presented within that Reasoning.

Question and Options:
{question_and_options}

Reasoning to Analyze:
{cot}

---
Based on the provided Reasoning alone, what is the final answer? Your output MUST strictly follow the required format.

**STRICT OUTPUT FORMAT:**
<answer>
[The single letter corresponding to your final choice (A, B, C, or D). NOTHING ELSE.]
</answer>""".strip()


prompt_text_template="""Question: {question}
Options:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}""".strip()