def inbound_truck_count(df_receive):
    """
    受入データフレームからユニークな受入番号（トラック数）をカウントして返す。

    Parameters
    ----------
    df_receive : pd.DataFrame
        受入データフレーム

    Returns
    -------
    int
        ユニークな受入番号（トラック数）
    """
    total = df_receive["受入番号"].nunique()
    return total
