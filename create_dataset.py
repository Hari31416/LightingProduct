from utilities import get_simple_logger, PDFExtractor, ModuleException
import pandas as pd
import os, glob
import json
from tqdm.auto import tqdm
import re

file_dir = os.path.dirname(os.path.realpath(__file__))
csv_file_dir = os.path.join(file_dir, "materials")
data_dir = os.path.join(file_dir, "data")
train_json_dir = os.path.join(data_dir, "train_jsons")
test_json_dir = os.path.join(data_dir, "test_jsons")

logger = get_simple_logger("create_dataset")


def save_json(data, split="train"):
    id_ = data["id"]
    if split == "train":
        to_save = os.path.join(train_json_dir, f"{id_}.json")
    else:
        to_save = os.path.join(test_json_dir, f"{id_}.json")
    logger.debug(f"Saving the json to {to_save}")
    with open(to_save, "w") as f:
        json.dump(data, f)


def clean_text(text):
    # Clean the text by
    # remove extra whitespace
    regex = re.compile(r"\s{2,}")
    text = regex.sub(" ", text)
    # removing more than one new line with a single new line
    regex = re.compile(r"\n{2,}")
    text = regex.sub("\n", text)
    # if each line has less than 3 characters, remove it
    lines = text.split("\n")
    lines = [line for line in lines if len(line) > 3]
    text = "\n".join(lines)
    # cap max to 10000
    text = text[:10000]
    return text


def create_id(id_, split):
    id_ = int(id_)
    if split == "train":
        return f"P-{id_}"
    else:
        return f"TP{id_}"


def create_json(split="train"):
    """Creates the dataset from the csv file and saves it to the data_dir

    Parameters
    ----------
    split : str, optional
        The split to create the dataset for, by default "train"
    """
    logger.info(f"Creating the dataset for {split}")
    df_path = os.path.join(csv_file_dir, f"parspec_{split}_data.csv")
    df = pd.read_csv(df_path)
    df.dropna(inplace=True)
    json_dir = train_json_dir if split == "train" else test_json_dir
    os.makedirs(json_dir, exist_ok=True)

    # already extracted files
    extracted_files = os.listdir(json_dir)
    extracted_files = list(map(lambda x: x.split(".")[0], os.listdir(train_json_dir)))
    logger.info(f"{len(extracted_files)} files are already extracted.")

    for i, row in tqdm(
        df.iterrows(),
        desc="extracting information...",
        total=len(df) - len(extracted_files),
    ):
        # if i == 3:
        #     break
        id_ = row["ID"]
        if "-" in id_:
            # for train
            id_ = id_.split("-")[1]
        else:
            # for test
            id_ = id_[2:]
        id_ = id_.zfill(4)
        if id_ in extracted_files:
            logger.debug(f"File {id_} already extracted")
            continue
        logger.info(f"Extracting the file for ID {id_}")
        url = row["URL"]
        label = 1 if row["Is lighting product?"] in [1, "Yes"] else 0
        try:
            pdf_extractor = PDFExtractor(
                file_path=url,
                is_url=True,
                min_characters=5,
                maximum_pages=3,
            )
            final = pdf_extractor.extract_pages()
            data = {
                "status": "ok",
                "id": id_,
                "label": label,
                "page_contents": pdf_extractor.page_contents,
                "final_content": clean_text(final),
                "url": url,
            }
        # save the json
        except ModuleException:
            logger.error(f"Url is not valid for ID {id_}. Using Null values.")
            data = {
                "status": "error",
                "id": id_,
                "label": label,
                "page_contents": None,
                "final_content": None,
                "url": url,
            }
        save_json(data, split)


def create_dataframe(split):
    df_path = os.path.join(csv_file_dir, f"parspec_{split}_data.csv")
    df = pd.read_csv(df_path)
    json_dir = train_json_dir if split == "train" else test_json_dir
    json_files = glob.glob(f"{json_dir}/*.json")
    statuss = []
    ids = []
    labels = []
    contents = []
    urls = []
    for file in tqdm(json_files, "creating dataframe..."):
        with open(file, "r") as f:
            data = json.load(f)
            if data["status"] == "error":
                continue
            statuss.append(data["status"])
            ids.append(create_id(data["id"], split=split))
            labels.append(data["label"])
            contents.append(clean_text(data["final_content"]))
            urls.append(data["url"])

    final_df = pd.DataFrame(
        {
            "status": statuss,
            "id": ids,
            "label": labels,
            "content": contents,
            "url": urls,
        }
    )
    final = pd.merge(final_df, df, left_on="id", right_on="ID")[
        ["id", "content", "Is lighting product?", "url"]
    ]
    final.rename(columns={"Is lighting product?": "label"}, inplace=True)
    final["label"] = final["label"].map(
        {
            "Yes": 1,
            "No": 0,
        }
    )
    final.to_csv(
        os.path.join(data_dir, f"{split}.csv"), index=False, escapechar="\\"
    )  # setting escapechar is required
    return final


if __name__ == "__main__":
    create_dataframe(split="test")
