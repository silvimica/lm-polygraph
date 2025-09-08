import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset as hf_dataset

from typing import Iterable, Tuple, List, Union, Optional


class Dataset:
    """
    Seq2seq dataset for calculating quality of uncertainty estimation method.
    """

    def __init__(self, x: List[str], y: List[str], batch_size: int):
        """
        Parameters:
            x (List[str]): a list of input texts.
            y (List[str]): a list of output (target) texts. Must have the same length as `x`.
            batch_size (int): the size of the texts batch.
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __iter__(self) -> Iterable[Tuple[List[str], List[str]]]:
        """
        Returns:
            Iterable[Tuple[List[str], List[str]]]: iterates over batches in dataset,
                returns list of input texts and list of corresponding output texts.
        """
        for i in range(0, len(self.x), self.batch_size):
            yield (
                self.x[i : i + self.batch_size],
                self.y[i : i + self.batch_size],
            )

    def __len__(self) -> int:
        """
        Returns:
            int: number of batches in the dataset.
        """
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def select(self, indices: List[int]):
        """
        Shrinks the dataset down to only texts with the specified index.

        Parameters:
            indices (List[int]): indices to left in the dataset.Must have the same length as input texts.
        """
        self.x = [self.x[i] for i in indices]
        self.y = [self.y[i] for i in indices]
        return self

    def train_test_split(self, test_size: int, seed: int, split: str = "train"):
        """
        Samples dataset into train and test parts.

        Parameters:
            test_size (int): size of test dataset,
            seed (int): seed to perform random splitting with,
            split (str): either 'train' or 'test'. If 'train', lefts only train data in the current dataset object.
                If 'test', left only test data. Default: 'train'.

        Returns:
            Tuple[List[str], List[str], List[str], List[str]]: train input and target texts list,
                test input and target texts list.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(self.x),
            np.array(self.y),
            test_size=test_size,
            random_state=seed,
        )

        if split == "train":
            self.x = X_train.tolist()
            self.y = y_train.tolist()
        else:
            self.x = X_test.tolist()
            self.y = y_test.tolist()

        return (
            X_train.tolist(),
            X_test.tolist(),
            y_train.tolist(),
            y_test.tolist(),
        )

    def subsample(self, size: int, seed: int):
        """
        Subsamples the dataset to the provided size.

        Parameters:
            size (int): size of the resulting dataset,
            seed (int): seed to perform random subsampling with.
        """
        np.random.seed(seed)
        if len(self.x) < size:
            indices = list(range(len(self.x)))
        else:
            if size < 1:
                size = int(size * len(self.x))
            indices = np.random.choice(len(self.x), size, replace=False)
        self.select(indices)

    @staticmethod
    def from_csv(
        csv_path: str,
        x_column: str,
        y_column: str,
        batch_size: int,
        prompt: str = "",
        **kwargs,
    ):
        """
        Creates the dataset from .CSV table.

        Parameters:
            csv_path (str): path to .csv table,
            x_column (str): name of column to take input texts from,
            y_column (str): name of column to take target texts from,
            batch_size (int): the size of the texts batch.
        """
        csv = pd.read_csv(csv_path)
        x = csv[x_column].tolist()
        y = csv[y_column].tolist()

        if len(prompt):
            x = [prompt.format(text=text) for text in x]

        return Dataset(x, y, batch_size)

    @staticmethod
    def load_hf_dataset(
        path: Union[str, List[str]],
        split: str,
        **kwargs,
    ):
        load_from_disk = kwargs.pop("load_from_disk", False)
        if load_from_disk:
            dataset_name = path
            dataset = hf_dataset.load_from_disk(path)
        elif isinstance(path, str):
            dataset_name = path
            dataset = load_dataset(path, split=split, **kwargs)
        else:
            dataset_name = path[0]
            dataset = load_dataset(*path, split=split, **kwargs)

        return dataset_name, dataset

    @staticmethod
    def from_datasets(
        dataset_path: Union[str, List[str]],
        x_column: str,
        y_column: str,
        batch_size: int,
        prompt: str = "",
        description: str = "",
        mmlu_max_subject_size: int = 100,   # kept for compatibility; unused for ProX
        n_shot: int = 0,
        few_shot_split: str = "validation", # ProX has validation/test; default few-shot from validation
        few_shot_prompt: Optional[str] = None,
        instruct: bool = False,
        split: str = "test",
        size: int = None,
        mmlu_lang: Optional[str] = None,    # <- use this to pick the language subset, e.g. "en", "fr", ...
        **kwargs,
    ):
        """
        Creates the dataset from Huggingface datasets.

        Parameters:
            dataset_path (str): HF path to dataset,
            x_column (str): name of column to take input texts from (ignored for MMLU-ProX),
            y_column (str): name of column to take target texts from (ignored for MMLU-ProX),
            batch_size (int): the size of the texts batch,
            prompt (str): prompt template to use for input texts (default: ''),
            description (str): optional description header per-instance,
            n_shot (int): number of few-shot exemplars to prepend,
            few_shot_split (str): split to draw few-shot examples from (default: 'validation' for ProX),
            few_shot_prompt (Optional[str]): separate few-shot formatting template when `instruct=True`,
            instruct (bool): whether to use instruction-style few-shot formatting,
            split (str): dataset split to take data from (default: 'test'),
            size (Optional[int]): size to subsample dataset to. If None, the full dataset split will be taken.
            mmlu_lang (Optional[str]): language code for MMLU-ProX subset (e.g., 'en', 'fr', 'zh', ...).
        """

        dataset_name = str(dataset_path)

        # -------------------------
        # Special handling: MMLU-ProX
        # -------------------------
        if "mmlu-prox" in dataset_name.lower():
            if not mmlu_lang:
                raise ValueError("For li-lab/MMLU-ProX you must provide `mmlu_lang` (e.g., 'en', 'fr', 'zh').")

            # ProX uses language subsets; pass `name=mmlu_lang`.
            # Columns (per HF viewer): question, option_0..option_9, answer (A..J), answer_index (0..9), category, cot_content, etc.
            # We ignore x_column/y_column here and build prompts ourselves.
            _, dataset = Dataset.load_hf_dataset(dataset_path, split, **kwargs)

            few_shot_dataset = None
            if n_shot > 0:
                _, few_shot_dataset = Dataset.load_hf_dataset(dataset_path, few_shot_split, **kwargs)
                if len(few_shot_dataset) < n_shot:
                    raise ValueError(
                        f"Requested {n_shot} few-shot examples but only {len(few_shot_dataset)} available in {few_shot_split}."
                    )

            if size is not None and size < len(dataset):
                dataset = dataset.select(range(size))

            # Map index -> letter for 10-way multiple choice
            letters = list("ABCDEFGHIJ")

            def extract_choices(inst):
                # Collect options in order; guaranteed option_0..option_9 exist
                return [inst[f"option_{i}"] for i in range(10)]

            # Build few-shot prefix text (single string reused for all instances)
            few_shot_prefix = ""
            if n_shot > 0:
                ids = np.random.choice(len(few_shot_dataset), n_shot, replace=False)
                few_shot_data = few_shot_dataset.select(ids)

                if instruct:
                    if few_shot_prompt is None:
                        raise AssertionError("`few_shot_prompt` must be provided when `instruct=True`.")
                    # instruction-flavored
                    few_shot_prefix = "Here are a few examples of questions and answers:\n\n"
                    for inst in few_shot_data:
                        choices = extract_choices(inst)
                        # Use provided few_shot_prompt template
                        few_shot_prefix += few_shot_prompt.format(
                            choices=choices,
                            question=str(inst["question"]).strip(),
                            answer=letters[int(inst["answer_index"])],
                            category=inst.get("category", "")
                        ) + "\n\n"
                    few_shot_prefix += "Now answer the following question in the same format:\n\n"
                else:
                    # plain format using `prompt` template
                    for inst in few_shot_data:
                        choices = extract_choices(inst)
                        few_shot_prefix += prompt.format(
                            choices=choices,
                            question=str(inst["question"]).strip(),
                            answer=letters[int(inst["answer_index"])],
                            category=inst.get("category", "")
                        ) + "\n"

            # Build final X/Y
            x, y = [], []
            for inst in dataset:
                choices = extract_choices(inst)

                # Format the evaluation prompt for this item
                # User-supplied `prompt` should expect {question}, {choices}, and {answer} (leave empty here)
                formatted_prompt = prompt.format(
                    choices=choices,
                    question=str(inst["question"]).strip(),
                    answer="",  # left blank for the model to fill
                    category=inst.get("category", "")
                )

                # Optional description/header
                prefix = ""
                if description:
                    # You can let the caller include things like language name in the description string if desired.
                    # e.g., description="Language: {lang} | Category: {category}"
                    # We support {lang} and {category} placeholders.
                    prefix = description.format(
                        lang=mmlu_lang,
                        category=inst.get("category", "")
                    ).strip()

                full_input = (
                    (prefix + "\n\n") if prefix else ""
                ) + (few_shot_prefix if few_shot_prefix else "") + formatted_prompt

                x.append(full_input)
                # Ground-truth is the letter (A–J). Prefer `answer_index` for numeric robustness.
                if "answer_index" in inst and inst["answer_index"] is not None:
                    y.append(letters[int(inst["answer_index"])])
                else:
                    # Fallback to `answer` (expected to be "A".."J")
                    y.append(str(inst["answer"]).strip())

            return Dataset(x, y, batch_size)

        # -------------------------
        # Your existing logic (unaltered)
        # -------------------------
        dataset_name, dataset = Dataset.load_hf_dataset(dataset_path, split, **kwargs)
        few_shot_dataset = None

        if n_shot > 0:
            _, few_shot_dataset = Dataset.load_hf_dataset(
                dataset_path, few_shot_split, **kwargs
            )

        if size is not None and size < len(dataset):
            dataset = dataset.select(range(size))

        if "allenai/c4" in dataset_name.lower():
            x, y = [], []
            for inst in dataset:
                if len(inst[x_column]) <= 1024:
                    x.append(inst[x_column])
                    y.append(inst[y_column])

        elif ("mmlu" in dataset_name.lower()) and mmlu_lang:
            answers = ["A", "B", "C", "D"]
            subjects = np.array(dataset["subject"])
            few_shot_subjects = np.array(few_shot_dataset["subject"]) if few_shot_dataset is not None else None
            x, y = [], []
            for subject in np.unique(subjects):
                formatted_description = description.format(
                    subject=subject.replace("_", " ")
                )
                formatted_few_shot_prompt = ""
                if n_shot > 0 and few_shot_dataset is not None:
                    few_shot_subject = few_shot_dataset.select(
                        np.argwhere(few_shot_subjects == subject).flatten()
                    )
                    few_shot_ids = np.random.choice(
                        len(few_shot_subject), n_shot, replace=False
                    )
                    few_shot_data = few_shot_subject.select(few_shot_ids)
                    if instruct:
                        assert (
                            few_shot_prompt is not None
                        ), "separate few_shot_prompt must be provided for instruction mode."
                        formatted_few_shot_prompt = (
                            "Here are a few examples of questions and answers:\n\n"
                        )
                        for inst in few_shot_data:
                            formatted_few_shot_prompt += (
                                few_shot_prompt.format(
                                    choices=inst["choices"],
                                    question=inst["question"].strip(),
                                    answer=answers[inst["answer"]],
                                )
                                + "\n\n"
                            )
                        formatted_few_shot_prompt += (
                            "Now answer the following question in the same format:\n\n"
                        )
                    else:
                        for inst in few_shot_data:
                            formatted_few_shot_prompt += (
                                prompt.format(
                                    choices=inst["choices"],
                                    question=inst["question"].strip(),
                                    answer=answers[inst["answer"]],
                                )
                                + "\n"
                            )

                subject_data = dataset.select(
                    np.argwhere(subjects == subject).flatten()
                )

                if len(subject_data) > mmlu_max_subject_size:
                    subject_data = subject_data.select(range(mmlu_max_subject_size))

                for inst in subject_data:
                    formatted_prompt = prompt.format(
                        choices=inst["choices"],
                        question=inst["question"].strip(),
                        answer="",
                    )
                    x.append(
                        formatted_description
                        + "\n\n"
                        + formatted_few_shot_prompt
                        + formatted_prompt
                    )
                    y.append(answers[inst[y_column]])

        else:
            x = dataset[x_column]
            if y_column is not None:
                y = dataset[y_column]
            else:
                y = ["" for _ in range(len(x))]

        return Dataset(x, y, batch_size)

    @staticmethod
    def load(path_or_path_and_files: Union[str, List[str]], *args, **kwargs):
        """
        Creates the dataset from either local .csv path (if such exists) or Huggingface datasets.
        See `from_csv` and `from_datasets` static functions for the description of *args and **kwargs arguments.

        Parameters:
            path_or_path_and_files (str or List[str]): local path to .csv table or HF path to dataset.
        """
        if isinstance(path_or_path_and_files, str) and os.path.isfile(
            path_or_path_and_files
        ):
            return Dataset.from_csv(path_or_path_and_files, *args, **kwargs)
        return Dataset.from_datasets(path_or_path_and_files, *args, **kwargs)
