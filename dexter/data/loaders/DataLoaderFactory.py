from dexter.config.constants import Dataset, Split
from dexter.data.loaders.MusiqueQaDataLoader import MusiqueQADataLoader
from dexter.data.loaders.WikiMultihopQADataLoader import WikiMultihopQADataLoader


class DataLoaderFactory:
    ''' Data Loader factory to map dataset alias to corresponding Data loader class'''


    def create_dataloader(
        self,
        dataloader_name: str,
        tokenizer="bert-base-uncased",
        config_path="test_config.ini",
        split=Split.TRAIN,
        batch_size=None,
        corpus=None
    ):
        if Dataset.WIKIMULTIHOPQA in dataloader_name:
            loader = WikiMultihopQADataLoader
        elif Dataset.MUSIQUEQA in dataloader_name:
            loader = MusiqueQADataLoader
        else:
            raise NotImplemented(f"{dataloader_name} not implemented yet.")
        return loader(dataset=dataloader_name, config_path=config_path,
                      split=split,batch_size=batch_size,
                      tokenizer=tokenizer,
                      corpus=corpus)