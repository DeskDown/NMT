from datasets import Dataset, DatasetDict
import os
import pandas as pd


class ALTDataset:

    def __init__(
        self,
        subset_languages,
        raw_file="NMT/YY.csv",
        default_split=True,
        random_state=99
    ):

        assert os.path.isfile(raw_file), "Check the raw_file path or Clone the github repository: \
            git clone https://github.com/DeskDown/NMT.git"

        self.raw_file = raw_file
        self.rs = random_state
        self.default_split = default_split
        self.subset_languages = subset_languages
        self.full_dataframe = None
        self.train_df, self.eval_df, self.test_df = self._init_dataframes()

    def _init_dataframes(self):
        yy_subset = self.subset_languages

        df = pd.read_csv(self.raw_file, sep="|")
        df.dropna(inplace=True)
        self.df = df.copy()
        df_en = df[df.lang == 'en'].copy()
        df_en.drop_duplicates(["SID"], inplace=True)
        dfs = []

        for other in yy_subset:
            df_other = df[df.lang == other].copy().drop_duplicates(["SID"])
            df_en_other = pd.merge(
                df_en, df_other, on='SID', suffixes=("_en", f"_{other}"))
            df_en_other.rename(columns={
                               f"Sent_{other}": "Sent_yy", f"lang_{other}": "lang_yy"}, errors="raise", inplace=True)
            dfs.append(df_en_other)

        df_all = dfs[0] if len(dfs) == 1 else pd.concat(
            dfs, axis=0, ignore_index=True)
        df_all["URL_id"] = df_all.SID.str.split(".").str[1]

        if not self.default_split:
            # Use a random split to make train,test,eval dataframes
            df_eval = df_all.groupby("lang_yy").sample(
                frac=0.2, random_state=self.rs)
            df_test = df_eval.groupby("lang_yy").sample(
                frac=0.5, random_state=self.rs).dropna()
            df_train = df_all.drop(df_eval.index).dropna()
            df_eval = df_eval.drop(df_test.index).dropna()
        else:
            # Use Standard Split provided by ALT Project
            train_ids, test_ids, eval_ids = self._get_default_splits()
            train_mask = df_all.URL_id.isin(train_ids)
            test_mask = df_all.URL_id.isin(test_ids)
            eval_mask = df_all.URL_id.isin(eval_ids)
            df_train = df_all[train_mask].dropna().drop(
                columns=['lang_en', 'URL_id'])
            df_test = df_all[test_mask].dropna().drop(
                columns=['lang_en', 'URL_id'])
            df_eval = df_all[eval_mask].dropna().drop(
                columns=['lang_en', 'URL_id'])

        return (df_train, df_eval, df_test)

    def _get_default_splits(self):
        train_split = 'https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-train.txt'
        test_split = 'https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-test.txt'
        eval_split = 'https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-dev.txt'

        train_ids = pd.read_csv(train_split, sep="\t", names=[
                                "id", "x"]).id.str.replace("URL.", "")
        test_ids = pd.read_csv(test_split, sep="\t", names=[
                               "id", "x"]).id.str.replace("URL.", "")
        eval_ids = pd.read_csv(eval_split, sep="\t", names=[
                               "id", "x"]).id.str.replace("URL.", "")

        return (train_ids, test_ids, eval_ids)


    def add_target(item):
        item["Sent_en"] = f'2{item["lang_yy"]} {item["Sent_en"]}'
        return item


    def get_datasets(self):
        train_ds = Dataset.from_pandas(
            ds.train_df).remove_columns("__index_level_0__")
        eval_ds = Dataset.from_pandas(
            ds.eval_df).remove_columns("__index_level_0__")
        test_ds = Dataset.from_pandas(
            ds.test_df).remove_columns("__index_level_0__")

        return DatasetDict({
            "train": train_ds,
            "test": test_ds,
            "eval": eval_ds
        }).map(add_target)


if __name__ == '__main__':
    ss = ['fil', 'vi', 'id', 'ms', 'ja', 'khm', 'th', 'hi', 'my', 'zh']
    file = r'./NMT/YY.csv'
    ds = ALTDataset(ss, file)
    alt_ds = ds.get_datasets()
