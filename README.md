# Charformer for Math

Currently, this repository contains a custom implementation of Gradient-Based Subword Tokenization from the Charformer Paper. 

However, the main purpose of the repo is to explore this question: How important is tokenization for language models to do mathematics? 
Intuitively, for example, a BPE-tokenizer might tokenize numbers (especially large ones) in an expression in terms of numbers frequently seen before, potentially hampering the ability to process them correctly or at least forcing the model to spend more resources to reverse this poor tokenization. 
For example, here's how GPT-2/3's tokenizer tokenizes this question from DeepMind's Mathematics Dataset: ![](https://github.com/girivad/charformer_for_math/blob/main/presentation/Final%20GPT3%20Tokenization%20Example.png)

Could a better tokenization allow a language model to more easily and successfully perform mathematics?

In order to test this, I intend to follow this general recipe:

Compare a language model (GPT-2, since obvious issues are visible with the GPT Tokenizer) zero-shot, few-shot (i.e. fed training examples in the prompt to teach formatting - not intrinsic mathematical ability), finetuned on math training data, and finetuned on math training data but coupled with a ![GBST](https://arxiv.org/pdf/2106.12672.pdf) layer that simultaneously learns how to tokenize best on the ![DeepMind mathematics dataset](https://github.com/google-deepmind/mathematics_dataset)

If it is true that GBST + LM outperforms other settings (and indeed tokenization is a major hamper on mathematics performance of LLMs), it would be further worthwhile to explore whether learning the GBST module on a general dataset (like some sample of the Pile) could still learn to tokenize appropriately from math, without impacting standard language modeling performance (i.e. learn how to tokenize from context). This could allow greater generalization across domains, and would be fascinating to see!

## Results [Pending]
