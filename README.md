# Charformer for Math

Currently, this repository contains a custom implementation of Gradient-Based Subword Tokenization from the Charformer Paper. 

However, the main purpose of the repo is to explore this question: How important is tokenization for language models to do mathematics? 
Intuitively, for example, a BPE-tokenizer might tokenize numbers (especially large ones) in an expression in terms of numbers frequently seen before, potentially hampering the ability to process them correctly or at least forcing the model to spend more resources to reverse this poor tokenization. Could a better, learned tokenizer provide better representations that allow a language model to more easily and successfully perform mathematics?

In order to test this, I intend to follow this general recipe:

Compare a language model (most likely GPT-2, due to the ease of using custom embeddings, and of finetuning with limited resources, though a Llama model would be much more relevant) zero-shot, few-shot (i.e. fed training examples in the prompt to teach formatting - intrinsic mathematical ability), finetuned on math training data, and finetuned on math training data but coupled with a GBST layer that simultaneously learns how to tokenize best. I'm using the DeepMind Mathematics Dataset, which contains hundreds of thousands of programmatically generated mathematics questions and answers. I'm particularly interested in the large number arithmetic task!

If it is true that GBST + LM outperforms other settings (and indeed tokenization is a major hamper on mathematics performance of LLMs), it would be further worthwhile to explore whether learning the GBST module on a general dataset (like some sample of the Pile) could still learn to tokenize appropriately from math, without impacting standard language modeling performance (i.e. learn how to tokenize from context). This could allow greater generalization across domains, and would be fascinating to see!

## Results [Pending]
