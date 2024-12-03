def _generate_question(sample: dict, answer: bool, add_full_example: bool, answer_letter) -> str:
    s = f"The following are multiple choice questions (with answers) about {sample['subject']}.\n"
    choices = sample['choices']
    s += f"{sample['question']}\n"
    s += f"A. {choices[0]}\n"
    s += f"B. {choices[1]}\n"
    s += f"C. {choices[2]}\n"
    s += f"D. {choices[3]}\n"
    s += f"Answer:"
    if answer:
        idx = sample['answer']
        letter = chr(ord('A') + idx)
        ans = letter if answer_letter else idx
        s += f" {ans}"
        if add_full_example:
            s += f". {choices[idx]}"
    return s


def create_prompt(sample: dict) -> str:
    """
    Generates a prompt for a multiple choice question based on the given sample.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.

    Returns:
        str: A formatted string prompt for the multiple choice question.
    """
    return _generate_question(sample, True, False, False)


def create_prompt_with_examples(sample: dict, examples: list, add_full_example: bool = False) -> str:
    """
    Generates a 5-shot prompt for a multiple choice question based on the given sample and examples.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.
        examples (list): A list of 5 example dictionaries from the dev set.
        add_full_example (bool): whether to add the full text of an answer option

    Returns:
        str: A formatted string prompt for the multiple choice question with 5 examples.
    """
    s = ""
    for question in examples:
        s += _generate_question(question, True, add_full_example, True) + '\n\n'
    s += _generate_question(sample, False, add_full_example, True)
    return s
