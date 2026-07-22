import pandas as pd
def dummy(
        index,
        start,
        end=None,
        dummy_type="pulse"):

    dummy = pd.Series(
        0,
        index=index
    )

    if dummy_type == "pulse":

        if end is None:
            end = start

        dummy.loc[start:end] = 1

    elif dummy_type == "step":

        dummy.loc[start:] = 1

    return dummy
