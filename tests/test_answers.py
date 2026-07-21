from huginn_research import answers


def test_numeric_extract_answer_tag():
    text = "Let's compute: 17 + 25 = 42. <answer>42</answer>"
    assert answers.numeric_extract_answer(text) == "42"


def test_numeric_extract_answer_boxed_last_wins():
    text = r"First I think \boxed{41}, but wait, let me redo it: \boxed{42}"
    assert answers.numeric_extract_answer(text) == "42"


def test_numeric_extract_answer_thousands_normalization():
    item = {"question": "q", "answer": "1.300"}
    assert answers.numeric_process_item(item)["answer"] == "1300"
    assert answers.numeric_process_item({"question": "q", "answer": "1.3"})["answer"] == "1.3"


def test_numeric_extract_answer_missing():
    assert answers.numeric_extract_answer("") == "no_final_answer"
    assert answers.numeric_extract_answer("I have no idea.") == "no_final_answer"


def test_numeric_process_item_shape():
    item = {"id": "1", "question": "What is 17 + 25?", "answer": "42"}
    normalized = answers.numeric_process_item(item)
    assert normalized == {"id": "1", "question": "What is 17 + 25?", "answer": "42"}


def test_mc_extract_answer_tag():
    text = "The reasoning suggests option B. <answer>B</answer>"
    assert answers.mc_extract_answer(text) == "B"


def test_mc_extract_answer_last_wins():
    text = "Maybe <ans>A</ans> ... actually <ans>C</ans>"
    assert answers.mc_extract_answer(text) == "C"


def test_mc_extract_answer_boxed_number_mapped_to_letter():
    options = ["A) 10", "B) 20", "C) 30"]
    text = r"So the value is \boxed{20}"
    assert answers.mc_extract_answer(text, options) == "B"


def test_mc_extract_answer_missing():
    assert answers.mc_extract_answer("") == "no_final_answer"
    assert answers.mc_extract_answer("no letters here") == "no_final_answer"


def test_mc_process_item_formats_options():
    item = {
        "id": "1",
        "question": "Which is prime?\n# Answer option: ignored",
        "options": ["A) 4", "B) 7", "C) 9"],
        "answer": "b",
    }
    normalized = answers.mc_process_item(item)
    assert normalized["answer"] == "B"
    assert "Options:" in normalized["question"]
    assert "A) 4" in normalized["question"]
    assert "ignored" not in normalized["question"]


def test_check_correct_numeric_tolerance():
    assert answers.check_correct("numeric", "42.0", "42")
    assert not answers.check_correct("numeric", "43", "42")
    assert not answers.check_correct("numeric", "no_final_answer", "42")


def test_check_correct_mc_case_insensitive():
    assert answers.check_correct("multiple-choice", "b", "B")
    assert not answers.check_correct("multiple-choice", "A", "B")
