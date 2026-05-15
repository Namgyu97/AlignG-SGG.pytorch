## Learning Context-Conditioned Predicate Semantics via Prototype Feedback

### Overview
<img width="1774" height="827" alt="image" src="https://github.com/user-attachments/assets/1f0750b1-9f1c-4e54-8a59-f8d47f639033" />

---

### Abstract
In scene graph generation, a central challenge is modeling polysemous predicates whose meanings shift across contexts. Prior approaches address this issue by decomposing predicates into multiple static prototypes or retrieving semantically similar exemplars. However, these strategies keep predicate representations static and cannot reorganize semantics to reflect image-specific evidence, leading to systematic confusions in ambiguous contexts. We propose **AlignG**, which learns context-conditioned predicate semantics via prototype feedback. AlignG infers image-conditioned predicate semantics from the relations within each image and feeds the adapted semantics back to recalibrate relation representations. The learning objective anchors this adaptation to global semantic centers, preventing semantic drift while still allowing selective reorganization when the scene provides consistent relational cues. Experiments on VG-150 and GQA-200 show consistent improvements over state-of-the-art baselines, with F@100 improvements of +1.4 and +2.7 under SGDet. We further visualize per-image prototype similarity shifts and observe coherent context-dependent reorganization where prototypes selectively merge or separate predicates according to scene evidence.

---
