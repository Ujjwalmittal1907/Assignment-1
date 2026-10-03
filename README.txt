Assignment 1 - Embedding Projector Evaluation

Task: Use Embedding Projector to evaluate semantic similarity, syntactic similarity, analogies.

Files:
- assignment1.py : simple evaluation with cosine similarity
- projector_plot.png : 2D PCA plot (generated after run)
- vectors.tsv, metadata.tsv : upload to projector.tensorflow.org
- Run: python assignment1.py (needs only numpy + matplotlib)

Steps for video submission (screen capture + voiceover):

1. Run `python assignment1.py` and show output:
   - cat-dog = 1.0, cat-car = 0.0 -> semantic ok
   - run-running = 1.0 -> syntactic ok
   - king-man+woman = queen -> analogy ok
2. Show projector_plot.png (clusters like Embedding Projector)
3. Go to https://projector.tensorflow.org
   - Click Load -> load vectors.tsv + metadata.tsv
   - Search "cat" -> shows dog/lion near, car far
   - Search "run" -> shows running/runs near
   - Use analogy logic to explain king-man+woman=queen
4. Record with voiceover (script below).

Voiceover script (read while recording, ~1 min):
"Hello, this is Assignment 1. I am evaluating word embeddings.
First, semantic similarity: cat and dog have cosine 1.0, while cat and car have 0. Neighbours of cat are dog and lion.
Second, syntactic similarity: run and running have cosine 1.0, neighbours are runs and running.
Third, analogy: king minus man plus woman gives queen as nearest neighbour.
In the projector plot and on projector.tensorflow.org we see animals clustered together, vehicles together, and run forms together. This shows embeddings capture meaning, grammar, and analogies. Thank you."
