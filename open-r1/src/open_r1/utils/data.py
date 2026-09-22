import logging
import os
import datasets
from datasets import DatasetDict, concatenate_datasets

from ..configs import ScriptArguments


logger = logging.getLogger(__name__)


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    Args:
        args (ScriptArguments): Script arguments containing dataset configuration.

    Returns:
        DatasetDict: The loaded datasets.
    """
    if args.dataset_name and not args.dataset_mixture:
        # Expand ${SCRATCH} etc. so the local save_to_disk detection below
        # (os.path.isdir) sees the real path, not the literal env-var string.
        args.dataset_name = os.path.expandvars(args.dataset_name)
        logger.info(f"Loading dataset: {args.dataset_name}")
        # Local save_to_disk dirs have a `dataset_dict.json` at the top
        # (or `dataset_info.json` for single splits) — detect and use
        # load_from_disk to preserve all columns. Otherwise fall back
        # to the standard load_dataset path (HF hub or arrow/parquet
        # data dir).
        if (os.path.isdir(args.dataset_name) and
            (os.path.exists(os.path.join(args.dataset_name, "dataset_dict.json"))
             or os.path.exists(os.path.join(args.dataset_name, "dataset_info.json")))):
            logger.info("  using load_from_disk (local save_to_disk format)")
            return datasets.load_from_disk(args.dataset_name)
        return datasets.load_dataset(args.dataset_name, args.dataset_config)

    elif args.dataset_mixture:
        logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")
        seed = args.dataset_mixture.seed
        datasets_list = []

        for dataset_config in args.dataset_mixture.datasets:
            logger.info(f"Loading dataset for mixture: {dataset_config.id} (config: {dataset_config.config})")
            ds = datasets.load_dataset(
                dataset_config.id,
                dataset_config.config,
                split=dataset_config.split,
            )
            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)
            if dataset_config.weight is not None:
                ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
                logger.info(
                    f"Subsampled dataset '{dataset_config.id}' (config: {dataset_config.config}) with weight={dataset_config.weight} to {len(ds)} examples"
                )

            datasets_list.append(ds)

        if datasets_list:
            combined_dataset = concatenate_datasets(datasets_list)
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

            if args.dataset_mixture.test_split_size is not None:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size, seed=seed
                )
                logger.info(
                    f"Split dataset into train and test sets with test size: {args.dataset_mixture.test_split_size}"
                )
                return combined_dataset
            else:
                return DatasetDict({"train": combined_dataset})
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")
