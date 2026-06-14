---
name: grill-me
description: Interview the user relentlessly about a plan or design until reaching shared understanding, resolving each branch of the decision tree. Use when user wants to stress-test a plan, get grilled on their design, or mentions "grill me".
---

Interview me relentlessly about every aspect of this plan until we reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one-by-one. For each question, provide your recommended answer.

Ask the questions one at a time.

If a question can be answered by exploring the codebase, explore the codebase instead.

## Decision Recording

Keep track of all decisions reached during the interview process. At the end of the session, save the complete decision log to a file in the project directory for future reference.

Decision log should include:
- Question asked
- User's answer
- Rationale discussed
- Final decision/recommendation
- Dependencies on other decisions

<thinking_mode>interleaved</thinking_mode>
<max_thinking_length>16000</max_thinking_length>
