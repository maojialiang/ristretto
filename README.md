# Ristretto
### [Paper](https://arxiv.org/pdf/)  | [Code](https://github.com/maojialiang/ristretto) 

Ristretto is a powerful Vision Language Model (VLM) that supports inputs such as text, image, and video, and possesses strong abilities in understanding, reasoning, and generation.

# News
**[2024/03/31]** Paper released on [arXiv](https://arxiv.org/abs/).

**[2024/03/25]** We released the Ristretto API.

# Model

Code and Weight will be released after the company verifies.


|            Model            |    Date    |                                           API                                            |                     Note                     |
| :-------------------------: | :--------: | :-------------------------------------------------------------------------------------------: | :------------------------------------------: |
| Ristretto-4B | 2025.03.25 | [infer](./ristretto_api.py) |                  Qwen2.5-3B + SigLIP2-400M                  |


# How to Use?
```python
python ristretto_api.py
```

# Performance

|Benchmark|Ristretto-4B|
|:---:|:---:|
|MMBench-V1.1<sub>test</sub>|81.8|
|MMStar|62.6|
|MMMU<sub>val</sub>|49.1|
|MathVista<sub>testmini</sub>|67.9|
|HallusionBench<sub>avg</sub>|50.2|
|AI2D<sub>test</sub>|84.0|
|OCRBench|85.1|
|MMVet|61.8|
|Average|67.9|