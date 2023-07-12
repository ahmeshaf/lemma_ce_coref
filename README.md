# $2*n$ is better than $n^2$: Decomposing Event Coreference Resolution into Two Tractable Problems
Accompanying code for the paper [2*n _is better than_ n^2: _Decomposing Event Coreference Resolution into Two Tractable Problems_](https://arxiv.org/abs/2305.05672) to be published in Findings of the Association of Computational Linguistics, ACL 2023.

Modelling code adapted from [aviclu/CDLM](https://github.com/aviclu/CDLM)

## Running paper's experiments
Check out the [notebook](PleaseRunThis.ipynb)

## Citation
Please use the following citation if you would like to use this work.
```
@inproceedings{ahmed-etal-2023-2,
    title = "$2*n$ is better than $n^2$: Decomposing Event Coreference Resolution into Two Tractable Problems",
    author = "Ahmed, Shafiuddin Rehan  and
      Nath, Abhijnan  and
      Martin, James H.  and
      Krishnaswamy, Nikhil",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.100",
    pages = "1569--1583",
    abstract = "Event Coreference Resolution (ECR) is the task of linking mentions of the same event either within or across documents. Most mention pairs are not coreferent, yet many that are coreferent can be identified through simple techniques such as lemma matching of the event triggers or the sentences in which they appear. Existing methods for training coreference systems sample from a largely skewed distribution, making it difficult for the algorithm to learn coreference beyond surface matching. Additionally, these methods are intractable because of the quadratic operations needed. To address these challenges, we break the problem of ECR into two parts: a) a heuristic to efficiently filter out a large number of non-coreferent pairs, and b) a training approach on a balanced set of coreferent and non-coreferent mention pairs. By following this approach, we show that we get comparable results to the state of the art on two popular ECR datasets while significantly reducing compute requirements. We also analyze the mention pairs that are {``}hard{''} to accurately classify as coreferent or non-coreferent.",
}
```
