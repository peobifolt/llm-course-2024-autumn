import torch


def generate_with_reward_guidance(
        main_model, main_tokenizer,
        reward_model, reward_tokenizer,
        N=16,
        device='cpu',
    ):
    """
    Generate text samples using a main model and select the best sample based on a reward model's guidance.

    This function generates multiple text samples from a main model, evaluates each sample using a reward model,
    and returns the sample with the highest reward score. The process is guided by the reward model to select
    the most desirable output.

    Parameters:
    main_model: The language model used to generate text samples.
    main_tokenizer: The tokenizer for main_model
    reward_model: The model used to compute reward scores for the generated samples.
    reward_tokenizer: The tokenizer for reward_model
    N (int, optional): The number of text samples to generate. Default is 16.
    device (str, optional): The device on which the computation should be performed. Default is 'cpu'.

    Returns:
    str: The generated text sample with the highest reward score.
    """
    generated_samples = []
    for _ in range(N):
        tokenized = main_tokenizer([''], return_tensors='pt')
        input_ids = tokenized['input_ids']
        generated_text_ids = main_model.generate(input_ids)
        generated_text = main_tokenizer.decode(generated_text_ids[0].tolist())
        generated_samples.append(generated_text)
    reward_scores = []
    for sample in generated_samples:
        if reward_model is None or reward_tokenizer is None:
            if sample.isnumeric():
                rewards = [int(sample)]
            else:
                rewards = [1]
        else:
            input_ids = reward_tokenizer(
                [sample], padding=True, truncation=True, return_tensors='pt'
            )
            input_ids = {key: value.to(device) for key, value in input_ids.items()}
            with torch.no_grad():
                logits = reward_model(**input_ids).logits
                rewards = logits[:, 0]
        reward_scores.append(rewards[0])
    best_sample_idx = reward_scores.index(max(reward_scores))
    best_sample = generated_samples[best_sample_idx]
    return best_sample
