from utils import class_names


def get_classnames(dataset: str):
    if dataset == "domainnet":
        return class_names.domainnet_classnames
    if dataset == "office31":
        return class_names.office31_classnames
    if dataset == "pacs":
        return class_names.pacs_classnames
    raise ValueError(f"Unsupported dataset: {dataset}")
