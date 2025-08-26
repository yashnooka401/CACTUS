## Align Environment and Agents with Expanded Tool Names in Processed Dataset

### Problem
Our processed dataset expands abstract tool labels (e.g., ResearchTool, SEOTool) into explicit, actionable tool names (e.g., Elicit, SciSpace, Semantic Scholar). However, the codebase (environment, agents, and preprocessing) may still mix coarse categories and fine-grained tool names for available tools, action space, and reward logic. This can cause agents and environments to select, reward, or process tools incorrectly.

### Required Changes

#### 1. Unified Tool Mapping
- Always use explicit tool names from the processed dataset for tool selection, rewards, and actions—never the original coarse category labels.

#### 2. Environment and Action Space
- Update environment logic (`MetaToolEnv.reset`, `utils.preprocess_dataset`) so `correct_tools` is always a list of explicit tool names (e.g., [Elicit, SciSpace, Semantic Scholar]), not generic labels.
- Example:
  ```python
  self.correct_tools = [t.strip() for t in row["tools"].split(",")]
  ```

#### 3. Agent Action and Vocabulary
- Build global tool vocabulary only from explicit tool names extracted from the processed dataset, not from abstract categories.
- Example:
  ```python
  tool_vocab = set()
  for tool_str in df["tools"].dropna():
      for t in tool_str.split(","):
          tool_vocab.add(t.strip())
  tool_vocab = sorted(tool_vocab)
  ```

#### 4. Reward Calculation
- Reward logic remains unchanged, but ensure `correct_tools` uses expanded tool names.

#### 5. Dataset Processing
- During preprocessing (in `utils.py`), always output lists of explicit tool names.
- Example:
  ```python
  def preprocess_dataset(df: pd.DataFrame) -> List[Dict]:
      processed = []
      for _, row in df.iterrows():
          tools = [t.strip() for t in row["tools"].split(",")]
          processed.append({
              "prompt": row["prompt"],
              "tools": tools
          })
      return processed
  ```

### Checklist
- [ ] Build global vocabulary from explicit tool names (all relevant files)
- [ ] Each query/task uses explicit tools as ground-truth (never abstract labels)
- [ ] Agent `select_action` and environment `step` operate only on concrete tools
- [ ] Preprocessing outputs lists of real tool names only

### References
- [config.py](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/82012496/b71d625d-0be1-455e-bfff-839da8e8d1fb/config.py)
- [metatool_env.py](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/82012496/dd3ac498-2a42-4681-9ae5-d7d49d8c6649/metatool_env.py)
- [utils.py](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/82012496/783867ba-6715-4db2-8148-f2b792cfa101/utils.py)
- [run_baseline_experiment.py](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/82012496/3a0bcfc4-d9fb-4c46-a423-63c90fd435c6/run_baseline_experiment.py)
- [baseline_agents.py](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/82012496/cf1087e5-c52f-4dac-ab65-b0742b488730/baseline_agents.py)