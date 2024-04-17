def prepare_metadata(metadata):
    for key in metadata:
        if isinstance(metadata[key], list):
            metadata[key] = ", ".join(map(str, metadata[key]))
        elif isinstance(metadata[key], dict):
            metadata[key] = prepare_metadata(metadata[key])
    return metadata
