import os
import pandas as pd
from transformers import DataProcessor, InputExample


class DementiaTalkBankProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].decode("utf-8"),  # tokenized text
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test_matched")

    def get_labels(self):
        """See base class."""
        labels = ["1.0", "0.0"]
        return [self.format(x) for x in labels]

    def format(self, x):
        return "_".join(x.lower().split())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            # text_b = line[9]
            label = None if set_type.startswith("test") else line[2]
            label = self.format(label)
            if label == "":
                label = "0.0"
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class MoralFoundationProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].decode("utf-8"),  # tokenized text
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test_matched")

    def get_labels(self):
        """See base class."""
        labels = ["fairness-cheating", "sanctity-degradation", "care-harm", "authority-subversion", "loyalty-betrayal"]
        return [self.format(x) for x in labels]

    def format(self, x):
        return "_".join(x.lower().split())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            # text_b = line[9]
            label = None if set_type.startswith("test") else line[2]
            label = self.format(label)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


if __name__ == '__main__':
    processor = DementiaTalkBankProcessor()
    examples = processor.get_dev_examples("/home/rohola/codes/dementia_analysis/data/dementia/unbalanced")
    a = 1
