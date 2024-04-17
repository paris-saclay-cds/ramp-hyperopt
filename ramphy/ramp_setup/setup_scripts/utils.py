def prepare_metadata(metadata):
    """Function to change lists into strings in the metadata. Useful for injecting metadata lists into 
    code templates"""
    for key in metadata:
        if isinstance(metadata[key], list):
            metadata[key] = ", ".join(map(str, metadata[key]))
        elif isinstance(metadata[key], dict):
            metadata[key] = prepare_metadata(metadata[key])
    return metadata
