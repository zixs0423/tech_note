---
layout: default
---

# AI Note Prompt Template

Use this prompt when chatting with an AI and you want to capture what you learned as a note for this repo.

## The Prompt

```
I have a tech note repo (Jekyll site). When I learn something useful from our conversation, generate a markdown snippet I can add to `docs/notes/<topic>.md`.

**Structure:**
- Frontmatter: `---\nlayout: default\n---`
- Top-level `#` = topic category (e.g. LLM, Deployment, Statistics)
- `##` = subtopic, `###` = specific item/concept name
- Under each `###`, always include these three `####` sections:
  1. **Concepts** — explanation in bullet points or short paragraphs. Use LaTeX for math.
  2. **Source** — links or references where this was learned
  3. **Code** — code snippets with language-tagged fences (if applicable; write "N/A" if none)
- Keep it concise and practical
- Output only the markdown, no extra commentary
```

## Example Output

```markdown
---
layout: default
---

# LLM

## Training

### RLHF

#### Concepts

- Reinforcement Learning from Human Feedback
- Train a reward model from human preference data
- Use PPO to optimize the policy against the reward model

#### Source

- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)

#### Code

```python
# PPO training loop (simplified)
for batch in dataloader:
    reward = reward_model(batch)
    policy_loss = -reward.mean()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
```
```
