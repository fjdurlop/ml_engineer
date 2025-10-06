
# Notes, presentation

task_2_notes.md

what is the task asking?
You need to benchmark ASR models across different dialects and measure WER vs latency trade-offs.

**Performance Metrics:**
- WER per dialect and overall
- Character Error Rate (CER)
- Real-time factor (RTF)
- 95th percentile latency


done:
- generate the results simulating the predicted text as no weights for the model
- metrics are computed

Missing:
- trade-offs analysis - check paper how
  - how:

Questions to prepare?
- how to do the dynamic selection?
  - automated testing (if D Variable is significantly influence by the Independent Variable ), check which ones are in the region of the xy graph with WER and latency
- Specific optimizations for worst-performing cases
  - fallbacks
  - more training data
- Trade-off analysis: accuracy vs speed vs memory
  - According to the use case

## from task to solution: what was added
- EvalConfig
- statistical testing moved to the other scripts
  - across models
  - across dialect
- some problems in forward pass of strides
## 1 normal

## sig differences accross models

## sig differences accross dialects