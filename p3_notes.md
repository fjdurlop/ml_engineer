
# Notes, presentation


what is the task asking?
Design and implement a streaming ASR system with continuous batching support for real-time speech recognition.

## Challenge Requirements
- Process audio chunks in real-time (streaming)
- Support multiple concurrent streams
- Implement dynamic batching across streams
- Maintain context across chunks per stream
- Handle variable-length inputs efficiently



done:
- processing loop
  - get chunk from stream
  - add to batch manager
  - get batch from batch manager
  - inference
    - simulating using functions from 2

Missing:


Questions to prepare?


